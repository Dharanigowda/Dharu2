FROM ubuntu
FROM python
# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
COPY . /app
CMD python /app/model.py && python /app/server.py