FROM python:3.8.9
COPY requirements.txt /app/
COPY LSTM.h5 /app/
COPY main.py /app/
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python","main.py"]



