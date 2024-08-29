from tensorflow/tensorflow:latest
WORKDIR /app
COPY mymodel.py /app/mymodel.py
CMD [ "python", "/app/mymodel.py" ]