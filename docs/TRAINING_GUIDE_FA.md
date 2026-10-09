# راهنمای آموزش چشم اختصاصی AvaVision

هدف: یک مدل ظاهر قرص که **مال خودتان** است و روی داده‌ی داروخانه‌ی خودتان آموزش دیده، جایگزین چشم پیش‌فرض Apple شود. کد در پوشه‌ی `training/` است. کل مسیر در CI روی Mac واقعی تست می‌شود (`.github/workflows/training.yml`).

## پیش‌نیاز

- Mac با Apple Silicon (برای آموزش از GPU مک استفاده می‌شود) یا یک سرور GPU اجاره‌ای
- Python 3.12 و Xcode

```bash
cd training
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## مراحل

### ۱. جمع کردن داده

در اپ، مغز را با این دو راه تغذیه کنید:

- آموزش هر دارو (Brain ← Teach a medication)
- بررسی‌های روزانه با امضای داروساز

بعد:

- Brain ← **Export training data** را بزنید.
- پوشه‌ی `AvaVision-training-data` را از اپ Files (On My iPhone ← AvaVision) به Mac منتقل کنید (AirDrop یا Finder).

برای امتحان مسیر بدون داده‌ی واقعی:

```bash
python tools/make_synthetic_export.py /tmp/export
```

### ۲. آموزش

```bash
python finetune.py --data AvaVision-training-data --out runs/v1
```

- **پایه:** DINOv2 ViT-S/14 with registers با مجوز Apache-2.0، آزاد برای استفاده‌ی تجاری.
- **روش:** Metric learning با Sub-center ArcFace. فقط دو بلوک آخر مدل پایه و یک لایه‌ی خروجی ۲۵۶ بعدی آموزش می‌بینند.
- **ارزیابی صادقانه:**
  - عکس‌ها و بررسی‌هایی که در آموزش نبوده‌اند
  - یک دارو که مدل اصلاً ندیده، برای سنجش رد کردن قرص ناشناخته
- **خروجی هر دوره:**
  - `nearest_neighbour_accuracy`: دقت شناسایی
  - `coverage_at_that_threshold`: پوشش، با آستانه‌ای که همه‌ی قرص‌های ناشناخته را رد می‌کند

### ۳. تبدیل به Core ML

```bash
python export_coreml.py --checkpoint runs/v1/embedder.pt --out build/AvaVisionEmbedder.mlpackage \
    --embedder-id avavision-dinov2s-v1
```

- روی Mac، خروجی Core ML با PyTorch مقایسه می‌شود. اگر شباهت کمتر از ۰٫۹۹ باشد، مدل رد می‌شود.
- اندازه‌ی مدل حدود ۴۲ مگابایت (FP16) است.

### ۴. نصب در اپ

```bash
../scripts/package-embedder.sh build/AvaVisionEmbedder.mlpackage avavision-dinov2s-v1 2026.10.1
cd ../App && xcodegen generate
```

این اسکریپت:

- مدل را کامپایل می‌کند.
- SHA-256 آن را می‌سازد.
- فایل‌های `AvaVisionEmbedder.mlmodelc` و `AvaVisionEmbedder.json` را در `App/AvaVision/Models/` می‌گذارد.

اگر hash نخواند، اپ مدل را استفاده نمی‌کند و به چشم Apple برمی‌گردد.

### ۵. بعد از نصب

- اپ با اولین اجرا، همه‌ی قرص‌های حافظه را با چشم جدید دوباره یاد می‌گیرد.
- آستانه‌ها دوباره کالیبره می‌شوند.
- اعتماد هر دارو از صفر ساخته می‌شود. این کار عمدی است: چشم جدید باید خودش را ثابت کند.

## پیشنهاد زمان‌بندی

| حجم داده‌ی تأییدشده | کار |
|---|---|
| تا ۱۰۰۰ قرص | چشم Apple کافی است؛ بیشتر آموزش دارو و برچسب بزنید |
| ۱۰۰۰ تا ۵۰۰۰ قرص | اولین DINOv2 تنظیم‌شده؛ مقایسه‌ی دقت با `brain-report` |
| بیش از ۵۰۰۰ قرص | آموزش دوره‌ای (مثلاً ماهانه) و آموزش مدل تشخیص اختصاصی (RF-DETR) |

## مدل تشخیص اختصاصی (مرحله‌ی بعد)

- **پیشنهاد تحقیق:** RF-DETR در اندازه‌های Nano، Small یا Medium (Apache-2.0)، با خروجی رسمی Core ML.
- **اجتناب کنید از:**
  - اندازه‌های Atto، Femto، Pico، XL و 2XL که مجوز PML-1.0 دارند
  - Ultralytics/YOLO با مجوز AGPL
- **داده:** عکس پک روی ایستگاه، با برچسب بیمار پوشانده‌شده، طبق [پروتکل داده](DATA_PROTOCOL_FA.md). برچسب‌زدن کادرها با کمک SAM 2.1 (Apache-2.0).

## قوانین مجوز

فقط وزن‌ها و کتابخانه‌های آزاد برای استفاده‌ی تجاری مجاز هستند.

| مدل | وضعیت |
|---|---|
| DINOv2 | مجاز (Apache-2.0) |
| SigLIP 2 | مجاز (Apache-2.0) |
| SAM 2/2.1 | مجاز (Apache-2.0) |
| RF-DETR N/S/M | مجاز (Apache-2.0) |
| DINOv3 | مجوز اختصاصی Meta؛ فقط پس از بررسی حقوقی |
| SAM 3 | مجوز اختصاصی؛ استفاده نکنید |
| MobileCLIP، ConvNeXt V2 (وزن‌ها) | فقط تحقیقاتی یا غیرتجاری؛ ممنوع |
| Ultralytics/YOLO، DEIMv2 | AGPL یا غیرتجاری؛ ممنوع |
