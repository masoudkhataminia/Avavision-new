# AvaVision

دستیار بررسی Webster-pak برای داروساز. اپ iOS هر خانه‌ی پک را با پروفایل دارویی مورد انتظار مقایسه می‌کند و موارد Missing، Extra، Wrong، Broken و Foreign را با مدرک قابل ممیزی نشان می‌دهد. **تصمیم نهایی همیشه با داروساز است.**

> این مخزن بازنویسی کامل AvaVision از صفر است و هیچ کدی از نسخه‌ی قبلی (fork اپ Ultralytics با مجوز AGPL) در آن نیست.
> **تمام حقوق محفوظ** ([LICENSE](LICENSE)). کد هیچ وابستگی خارجی ندارد.

## مغز یادگیرنده

اپ یک «مغز» دارد که ظاهر هر دارو را از تأییدهای داروساز یاد می‌گیرد و با هر بررسی قوی‌تر می‌شود:

- **از روز اول ظرفیت دارد:**
  - شمارش بدون آموزش با Apple Vision، کنترل‌شده با یک روش کلاسیک مستقل
  - چشم ظاهر Apple
  - آموزش هر دارو با چند عکس
- **قوی‌تر می‌شود:** هر خانه‌ای که داروساز تأیید می‌کند به حافظه اضافه می‌شود، آستانه‌ها خودکار کالیبره می‌شوند، و سابقه‌ی هر دارو ساخته می‌شود.
- **ایمن است:** مغز از روز اول فقط هشدار می‌دهد. قبول خودکار هر دارو را فقط با سابقه‌ی طولانی و بی‌خطا به دست می‌آورد، و با اولین خطا از دست می‌دهد.
- **چشم اختصاصی:** با `training/`، DINOv2 روی داده‌ی خود داروخانه آموزش می‌بیند و جایگزین چشم Apple می‌شود.

جزئیات: [مغز AvaVision](docs/BRAIN_FA.md) و [خلاصه‌ی تحقیق تکنولوژی روز](docs/RESEARCH_FA.md).

## وضعیت فعلی

| بخش | وضعیت |
|---|---|
| هسته‌ی منطق (`AvaVisionCore`) | کامل، با تست (روی Linux و macOS) |
| کنترل کیفیت تصویر (`AvaVisionImaging`) | کامل، با تست؛ آستانه‌ها هنوز روی ایستگاه واقعی کالیبره نشده‌اند |
| اپ iOS | کامل: پروفایل، دوربین/عکس، نتیجه، امضای داروساز، Audit |
| CI | GitHub Actions: Linux (lint + تست) و macOS (build + تست اپ در Simulator) |
| مغز یادگیرنده | کامل، با تست: حافظه، شناسایی open-set، کالیبراسیون خودکار، دفتر اعتماد، spot check |
| شمارش روز اول | segmenter داخلی (Apple Vision + کلاسیک)، روی Simulator تست شد؛ سرعت روی iPhone باید اندازه‌گیری شود |
| آموزش چشم اختصاصی | `training/` (DINOv2 → Core ML)، کل مسیر در CI روی Mac تست می‌شود |
| مدل تشخیص اختصاصی | هنوز آموزش ندیده (نیاز به عکس پک واقعی) |
| کالیبراسیون Layout | **انجام نشده.** تا کالیبره نشود هیچ خانه‌ای خودکار قبول نمی‌شود |

## ساختار

```
Package.swift                 Swift Package: هسته + ابزار خط فرمان
Sources/AvaVisionCore/        منطق دامنه: layout، پروفایل، هندسه، موتور تصمیم، Gate، Audit
Sources/AvaVisionImaging/     کیفیت تصویر: وضوح، نور، Glare
Sources/AvaVisionCore/Brain/  مغز یادگیرنده
Sources/AvaVisionCLI/         ابزار `avavision`: hash مدل، ارزیابی Holdout، بررسی Audit، گزارش مغز
training/                     آموزش آفلاین چشم اختصاصی (Python)
Tests/                        تست‌های هسته (روی Linux و Mac اجرا می‌شوند)
App/project.yml               تعریف پروژه‌ی Xcode (با XcodeGen ساخته می‌شود)
App/AvaVision/                اپ SwiftUI
App/AvaVisionTests/           تست‌های اپ (در Simulator)
docs/                         مستندات فارسی
scripts/ios-test.sh           ساخت و تست اپ روی Simulator
```

## شروع سریع (روی Mac)

پیش‌نیاز: Xcode 16 یا جدیدتر و [Homebrew](https://brew.sh).

```bash
brew install xcodegen
git clone https://github.com/masoudkhataminia/Avavision-new.git
cd Avavision-new/App
xcodegen generate
open AvaVision.xcodeproj
```

در Xcode، scheme `AvaVision` را انتخاب و Run کنید. راهنمای کامل تست روی Simulator و iPhone واقعی: [`docs/TESTING_GUIDE_FA.md`](docs/TESTING_GUIDE_FA.md).

## تست‌ها

```bash
swift test               # تست‌های هسته (Mac یا Linux)
scripts/ios-test.sh      # ساخت و تست اپ روی Simulator (فقط Mac)
```

## مستندات

1. [معماری](docs/ARCHITECTURE_FA.md)
2. [قوانین ایمنی](docs/SAFETY_RULES_FA.md)
3. [راهنمای تست](docs/TESTING_GUIDE_FA.md)
4. [پروتکل جمع‌آوری داده](docs/DATA_PROTOCOL_FA.md)
5. [راهنمای مدل](docs/MODEL_GUIDE_FA.md)
6. [مغز AvaVision](docs/BRAIN_FA.md)
7. [راهنمای آموزش چشم اختصاصی](docs/TRAINING_GUIDE_FA.md)
8. [خلاصه‌ی تحقیق تکنولوژی روز](docs/RESEARCH_FA.md)
9. [نقشه‌ی راه تا محصول تجاری](docs/ROADMAP_FA.md)
10. [دفتر تصمیم‌ها](docs/DECISIONS_FA.md)

## قوانین ثابت

- داده‌ی بیمار، رمز و کلید هرگز وارد Git نمی‌شود. اپ عکس پک ذخیره نمی‌کند (فقط hash آن در Audit)؛ مغز فقط تصویر تک‌قرص نگه می‌دارد.
- مدرک ضعیف، ناقص یا متناقض هرگز «قبول» نمی‌شود؛ به بررسی داروساز می‌رود.
- `Verified` فقط با Layout کالیبره‌شده و یکی از این دو ممکن است: مدل `released` که از Gate رد شده، یا اعتماد کسب‌شده‌ی مغز برای همه‌ی داروهای آن خانه.
- هر پک قبل از خروج، امضای داروساز لازم دارد و در Audit زنجیره‌ای ثبت می‌شود.
