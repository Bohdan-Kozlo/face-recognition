# Готові моделі для локальної перевірки облич

Дата перевірки: 2026-08-21. Нижче — моделі, що вже навчені на face-recognition
datasets і можуть створювати embedding без навчання на CelebA. Це дослідження,
а не зміна коду проєкту.

## Рекомендація для цього навчального локального проєкту

Спробувати **InsightFace `buffalo_l`**. Це готовий ONNX-пакет: детектор
SCRFD-10GF і recognition-модель ResNet50@WebFace600K, яка повертає
512-вимірний embedding. Пакет є default в InsightFace; автори наводять для
нього LFW 99.83, CFP-FP 99.33 та AgeDB-30 98.23. Ці цифри — результати
авторів на benchmark-наборах, а не гарантія для власних фото.
[Model Zoo](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md)

Для порівняння двох фото не потрібне навчання: знайти одне обличчя на кожному
фото, отримати два embedding і порівняти їх cosine similarity. Поріг треба
налаштувати на власних validation-парах; старий поріг `0.2086` від ResNet18
цьому пакету не підходить, бо embedding-простір інший.

Мінімальний локальний приклад з офіційного API:

```python
import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
app.prepare(ctx_id=0)
face = app.get(cv2.imread("uploads/same_1.jpeg"))[0]
embedding = face.normed_embedding  # shape: [512]
```

На першому запуску пакет завантажує pretrained models; їх також можна
завантажити вручну до `~/.insightface/models/`.
[Python package README](https://github.com/deepinsight/insightface/blob/master/python-package/README.md)

## Інші придатні варіанти

| Варіант | Що готове | Коли обрати |
|---|---|---|
| **InsightFace `buffalo_l`** | Детекція, landmarks, alignment і 512-D recognition embedding в одному пакеті. Приблизний розмір пакета 326 MB. | Найкращий перший кандидат для якості й простого локального inference. |
| InsightFace `buffalo_sc` | Легкий 16 MB пакет з MBF recognition-моделлю та SCRFD-500MF detector. | Якщо важливий невеликий розмір/CPU, але якість нижча за `buffalo_l` за таблицею авторів. |
| InsightFace `antelopev2` | Важчий пакет: SCRFD-10GF + ResNet100@Glint360K, близько 407 MB. | Лише як наступний експеримент, якщо GPU/диск не проблема. |
| `facenet-pytorch` `InceptionResnetV1(pretrained="vggface2")` | MTCNN для crop + 512-D L2-normalized embedding; вага автоматично кешується. | Якщо потрібен чистий PyTorch pipeline замість ONNX Runtime. Модель очікує crop 160x160. |
| DeepFace | Обгортка з `verify`, `represent`, `find`; може вибрати `ArcFace`, `Facenet`, `Facenet512`, `SFace`, `GhostFaceNet`, `Buffalo_L` та інші backends. | Для швидкого експерименту з готовим API, але це важча залежність і не окрема модель. |

InsightFace описує склад кожного пакета, розмір і свої benchmark-результати у
[Model Zoo](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md).

`facenet-pytorch` офіційно надає готові ваги InceptionResnetV1, навчені на
VGGFace2 або CASIA-Webface. Обидва повертають 512-D embedding; автори вказують
160x160 face crop та автоматичний download/cache ваг.
[facenet-pytorch README](https://github.com/timesler/facenet-pytorch/blob/master/README.md)

DeepFace не покращує модель сам по собі: це Python-обгортка, яка завантажує та
викликає різні зовнішні моделі. Її API дозволяє одразу перевірити дві фотографії:

```python
from deepface import DeepFace

result = DeepFace.verify(
    img1_path="uploads/same_1.jpeg",
    img2_path="uploads/same_2.jpeg",
    model_name="ArcFace",
    detector_backend="yunet",
    align=True,
)
```

[DeepFace README](https://github.com/serengil/deepface/blob/master/README.md)

## Ліцензії: важливе обмеження

`insightface` як код має MIT license, але **публічні pretrained model packs**
InsightFace, включно з `buffalo_l`, дозволені лише для некомерційного
дослідження. Для комерційного застосунку потрібна окрема ліцензія від
InsightFace. Отже `buffalo_l` придатний для вашого локального навчального
проєкту, але не можна просто перенести його в комерційний продукт.
[InsightFace licensing](https://github.com/deepinsight/insightface/blob/master/server/LICENSING.md)

DeepFace має MIT license лише на власний код. Його README прямо попереджає, що
ліцензії підключених моделей успадковуються і їх треба перевіряти перед
production-використанням. Це також стосується вибору `Buffalo_L` через
DeepFace: обгортка не скасовує ліцензію моделі.
[DeepFace license notice](https://github.com/serengil/deepface/blob/master/README.md#license)

Для `facenet-pytorch` варто окремо перевірити права на конкретні pretrained
ваги та датасет, на якому вони навчені, перед розгортанням. Його README
підтверджує походження ваг (VGGFace2/CASIA-Webface), але це не є юридичним
дозволом на будь-яке production-використання.

## Практичний наступний крок

Не замінювати поточну модель одразу. Зробити окрему локальну перевірку на тих
самих трьох фото (`same_1`, `same_2`, `different`) з `buffalo_l`; порівняти
similarity для same/different і тільки потім вирішити, чи інтегрувати пакет у
`src/`. Для справжньої авторизації також потрібні якісні фото, поріг, підібраний
на власних даних, та перевірка на підміну фотографією (liveness/anti-spoofing),
яких сам embedding не забезпечує.
