# راهنمای مدل

> **بدون مدل هم اپ می‌شمارد:** segmenter داخلی (Apple Vision + روش کلاسیک) با وضعیت count-only کار می‌کند. این راهنما برای مدل تشخیص اختصاصی است. چشم مغز در [راهنمای آموزش](TRAINING_GUIDE_FA.md) آمده.

اپ به هیچ مدل یا کتابخانه‌ی خاصی وابسته نیست. هر مدل **Core ML Object Detection** که خروجی `VNRecognizedObjectObservation` بدهد کار می‌کند (Create ML، یا مدل‌های دیگر با NMS داخلی).

## انتخاب ابزار آموزش و مجوز

| گزینه | مجوز | نکته |
|---|---|---|
| **Create ML** (اپ رایگان اپل روی Mac) | بدون محدودیت تجاری | ساده‌ترین شروع؛ خروجی مستقیم Core ML. پیشنهاد برای نسخه‌ی اول |
| **RF-DETR Nano/Small/Medium** | Apache-2.0 | پیشنهاد تحقیق؛ خروجی رسمی Core ML (`rfdetr[coreml]`). اندازه‌های Atto/Femto/Pico/XL/2XL مجوز PML-1.0 دارند و ممنوع‌اند |
| Ultralytics YOLO، DEIMv2 | AGPL-3.0 / غیرتجاری | ممنوع (D-116) |
| RT-DETRv2، D-FINE | Apache-2.0 | ممکن، ولی تبدیل به Core ML دستی است |

## ساخت مدل با Create ML (خلاصه)

۱. دیتاست را طبق [پروتکل داده](DATA_PROTOCOL_FA.md) با برچسب‌های `pill`، `broken`، `foreign` (و کد داروها برای هویت) آماده کنید.
۲. Create ML ← Object Detection ← داده‌ی آموزش را بدهید ← Train.
۳. خروجی `.mlmodel` را کامپایل کنید:

```bash
xcrun coremlcompiler compile Detector.mlmodel out/
mv out/Detector.mlmodelc out/AvaVisionDetector.mlmodelc
```

## Manifest

فایل `AvaVisionDetector.json` کنار مدل:

```json
{
  "schemaVersion": 1,
  "modelID": "avavision-detector",
  "version": "2026.11.0",
  "stage": "development",
  "modelSHA256": "<خروجی دستور hash>",
  "labels": {
    "pill": "pill",
    "broken": "broken",
    "foreign": "foreign",
    "metformin-500": "medication:metformin-500"
  },
  "notes": "Count-only pilot model"
}
```

```bash
swift run avavision hash out/AvaVisionDetector.mlmodelc     # مقدار modelSHA256
swift run avavision gate AvaVisionDetector.json out/AvaVisionDetector.mlmodelc   # ببینید Gate چه اجازه‌ای می‌دهد
```

## Stage و چرخه‌ی انتشار

| stage | اپ چه می‌کند |
|---|---|
| `development` | فقط شمارش |
| `validation` | فقط شمارش (در حال ارزیابی روی Holdout) |
| `released` | هویت، فقط اگر `evaluation` همه‌ی شرط‌های Gate را پاس کند |
| `revoked` | مدل اجرا نمی‌شود |

برای `released`: خروجی `avavision evaluate` را در فیلد `evaluation` بگذارید، stage را `released` کنید، hash را دوباره بررسی کنید.

## افزودن به اپ

۱. `AvaVisionDetector.mlmodelc` و `AvaVisionDetector.json` را در `App/AvaVision/Models/` کپی کنید (وارد Git نمی‌شوند).
۲. `cd App && xcodegen generate` و دوباره Build.
۳. در اپ: Settings ← Detection model باید stage و توانایی را نشان دهد. اگر hash نخواند، مدل اجرا نمی‌شود.

## نقطه‌ی کار مدل

- تشخیص‌های زیر ۲۵٪ اطمینان نویز حساب می‌شوند (`FrameAnalyzer.detectionNoiseFloor`).
- تشخیص‌های بین ۲۵٪ و ۶۰٪ شمرده نمی‌شوند و خانه را به Review می‌برند.
- ارزیابی Holdout باید با همین آستانه‌ها انجام شود.
