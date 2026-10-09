# راهنمای مدل تشخیص

> **بدون مدل هم ایستگاه می‌شمارد:** شمارش داخلی (دو روش مستقل OpenCV) با وضعیت count-only کار می‌کند. این راهنما برای مدل تشخیص اختصاصی است که بعد از جمع شدن عکس پک واقعی آموزش داده می‌شود. چشم مغز در [راهنمای آموزش](TRAINING_GUIDE_FA.md) آمده.

## انتخاب مدل و مجوز

| گزینه | مجوز | نکته |
|---|---|---|
| **RF-DETR Nano/Small/Medium** | Apache-2.0 | پیشنهاد اصلی؛ خروجی رسمی ONNX، دقیق روی اشیای کوچک، روی RTX در چند میلی‌ثانیه |
| RT-DETRv2، D-FINE | Apache-2.0 | ممکن؛ خروجی ONNX باید با قالب پایین سازگار شود |
| RF-DETR Atto/Femto/Pico/XL/2XL | PML-1.0 | ممنوع |
| Ultralytics YOLO، DEIMv2 | AGPL-3.0 / غیرتجاری | ممنوع (D-116) |

## قالب ONNX که ایستگاه می‌خواند

خروجی استاندارد RF-DETR (`model.export()`):

- ورودی `input`: تصویر RGB مربعی (1, 3, S, S) با نرمال‌سازی ImageNet. S از خود مدل خوانده می‌شود.
- خروجی `dets`: (1, Q, 4) کادرها به‌صورت (cx, cy, w, h) نرمال‌شده.
- خروجی `labels`: (1, Q, C) logit هر کلاس.

تشخیص‌های زیر ۲۵٪ نویز حساب می‌شوند؛ بین ۲۵٪ و ۶۰٪ شمرده نمی‌شوند و خانه را به Review می‌برند.

## Manifest

فایل `detector.json` کنار `detector.onnx`:

```json
{
  "schema_version": 1,
  "model_id": "avavision-rfdetr",
  "version": "2026.11.0",
  "stage": "development",
  "model_sha256": "<خروجی دستور hash>",
  "classes": ["pill", "broken", "foreign"],
  "labels": {
    "pill": "pill",
    "broken": "broken",
    "foreign": "foreign"
  },
  "notes": "Count-only pilot model"
}
```

- `classes`: نام کلاس‌ها به ترتیب خروجی مدل.
- `labels`: معنی هر کلاس. برای هویت دارو: `"metformin-500": "medication:metformin-500"`. کلاسی که معنی ندارد «قرص» حساب می‌شود، نه نادیده.

```bash
avavision gate detector.json detector.onnx      # Gate چه اجازه‌ای می‌دهد؛ خروجی hash را هم نشان می‌دهد
```

## Stage و چرخه‌ی انتشار

| stage | ایستگاه چه می‌کند |
|---|---|
| `development` | فقط شمارش |
| `validation` | فقط شمارش (در حال ارزیابی روی Holdout) |
| `released` | هویت، فقط اگر `evaluation` همه‌ی شرط‌های Gate را پاس کند |
| `revoked` | مدل اجرا نمی‌شود |

برای `released`: گزارش ارزیابی Holdout را در فیلد `evaluation` بگذارید و stage را `released` کنید. شرط‌ها در [قوانین ایمنی](SAFETY_RULES_FA.md) آمده‌اند.

## نصب روی ایستگاه

۱. `detector.onnx` و `detector.json` را در پوشه‌ی `models` ایستگاه کپی کنید (`%LOCALAPPDATA%\AvaVision\models`). این فایل‌ها وارد Git نمی‌شوند.
۲. ایستگاه را دوباره باز کنید.
۳. اگر hash نخواند یا مدل `revoked` باشد، ایستگاه مدل را اجرا نمی‌کند، به شمارش داخلی برمی‌گردد و دلیل را در `/api/status` (فیلد `model.note`) نشان می‌دهد.

## سرعت

مدل تشخیص روی کل عکس اجرا می‌شود (نه هر خانه جدا). روی کارت RTX با TensorRT، RF-DETR Small حدود ۵–۱۰ میلی‌ثانیه برای هر فریم طول می‌کشد. اندازه‌گیری: `avavision benchmark`.
