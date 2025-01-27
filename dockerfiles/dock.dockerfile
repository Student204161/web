FROM python:3.11-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY re.txt re.txt
COPY pyscript.py pyscript.py
WORKDIR /
RUN pip install -r re.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "pyscript.py"]