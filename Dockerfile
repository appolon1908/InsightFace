FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /models && curl --fail --show-error --location --retry 3 --output /models/face_detection_yunet_2023mar.onnx https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/face_detection_yunet_2023mar.onnx && echo "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  /models/face_detection_yunet_2023mar.onnx" | sha256sum --check - && curl --fail --show-error --location --retry 3 --output /models/face_recognition_sface_2021dec.onnx https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_recognition_sface/face_recognition_sface_2021dec.onnx && echo "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79  /models/face_recognition_sface_2021dec.onnx" | sha256sum --check -
COPY licenses/models /models/licenses
COPY NOTICE.md /models/NOTICE.md
COPY app /app/app
ENV DETECTOR_MODEL=/models/face_detection_yunet_2023mar.onnx
ENV RECOGNIZER_MODEL=/models/face_recognition_sface_2021dec.onnx
EXPOSE 8080
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8080"]
