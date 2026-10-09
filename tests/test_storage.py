from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from avavision.brain.brain import Brain
from avavision.brain.identity import IdentityDecision, IdentityEvidence
from avavision.brain.learning import LabellingTask, PillSighting
from avavision.core.models import WEEKLY_7X4, cell
from avavision.core.session import CheckSession
from avavision.core.signoff import CompartmentReview, ReviewOutcome, SignOff, SignOffDecision
from avavision.storage.database import AuditChainBroken, Database
from conftest import CATALOG, LAYOUT, METFORMIN, ONE, engine, frames, full_pack, profile


def test_catalog_layouts_profiles_and_settings(tmp_path):
    db = Database(tmp_path)
    db.save_catalog(CATALOG)
    p = profile(ONE)
    db.save_profile(p)
    db.save_layout(WEEKLY_7X4.model_copy(update={"width_mm": 260, "is_calibrated": True}))
    db.save_setting("orientation", "automatic")
    db.close()

    db = Database(tmp_path)
    assert db.catalog() == CATALOG
    assert db.profiles() == [p] and db.profile(str(p.id)) == p
    assert db.layouts()[0].width_mm == 260 and db.setting("orientation") == "automatic" and db.setting("x", 5) == 5
    db.delete_profile(str(p.id))
    assert db.profiles() == []


def test_audit_chain_persists_and_refuses_tampering(tmp_path):
    db = Database(tmp_path)
    session = CheckSession(LAYOUT, profile(ONE), CATALOG)
    for f in frames(full_pack()):
        session.record(f)
    session.analyze(engine())
    reviews = [
        CompartmentReview(compartment=i, outcome=ReviewOutcome.CONFIRMED_CORRECT) for i in LAYOUT.all_compartments
    ]
    record = session.complete(
        SignOff(pharmacist="MK", decision=SignOffDecision.RELEASED, reviews=reviews), "0.2", "s1", None
    )
    db.append_audit(record)
    db.append_audit(record.model_copy(update={"app_version": "0.3"}))
    assert db.verify_audit() is None and [r.app_version for r in db.audit_records()] == ["0.2", "0.3"]

    raw = sqlite3.connect(tmp_path / "avavision.sqlite3")
    raw.execute("UPDATE audit SET payload = replace(payload, '\"MK\"', '\"ZZ\"') WHERE sequence = 0")
    raw.commit()
    raw.close()
    assert db.verify_audit().defect == "hashMismatch"
    with pytest.raises(AuditChainBroken):
        db.append_audit(record)


def test_brain_round_trip_with_tasks_and_crop_pruning(tmp_path, synthetic):
    db = Database(tmp_path)
    assert db.load_brain("x") is None
    brain = Brain(synthetic.embedder)
    for g in range(2):
        brain.teach(METFORMIN, [(synthetic.pill(METFORMIN), f"m{g}{i}.jpg") for i in range(6)], group_id=f"g{g}")
    sighting = PillSighting(
        compartment=cell(0, 0),
        vector=synthetic.pill(METFORMIN),
        embedder_id=synthetic.embedder,
        crop_file="t.jpg",
        identity=IdentityEvidence(
            embedder_id=synthetic.embedder, policy_version=0, decision=IdentityDecision.identified(METFORMIN)
        ),
    )
    brain.labelling_queue.append(
        LabellingTask(check_id="c", compartment_label="D1", expected={METFORMIN: 1}, sightings=[sighting], priority=0.5)
    )
    brain.ledger.observe(IdentityDecision.identified(METFORMIN), METFORMIN)
    db.save_brain(brain)
    first = brain.knowledge.exemplars[0].id
    brain.knowledge.remove(first)
    db.save_brain(brain)

    loaded = Database(tmp_path).load_brain(synthetic.embedder)
    assert {e.id for e in loaded.knowledge.exemplars} == {e.id for e in brain.knowledge.exemplars}
    original = {e.id: e.vector for e in brain.knowledge.exemplars}
    assert all(np.array_equal(e.vector, original[e.id]) for e in loaded.knowledge.exemplars)
    assert loaded.ledger == brain.ledger and loaded.identity_policy == brain.identity_policy
    task = loaded.labelling_queue[0]
    assert task.sightings[0].identity.decision.named == METFORMIN and np.array_equal(
        task.sightings[0].vector, sighting.vector
    )

    for name in ("t.jpg", "stale.jpg", "m00.jpg"):
        (db.crops / name).write_bytes(b"x")
    db.prune_crops(db.referenced_crops(loaded))
    assert sorted(p.name for p in db.crops.iterdir()) == sorted({"t.jpg", "m00.jpg"} & db.referenced_crops(loaded))
