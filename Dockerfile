FROM ubuntu:20.04

RUN apt-get update && apt-get install -y build-essential


# Curix user
COPY . .

# switch to root
USER root


RUN apt-get update && apt-get install -y $(cat ubuntu_req.txt)
RUN apt-get -y install python3-pip 
RUN pip3 install -r requirements.txt
RUN apt-get autoremove -y

RUN mkdir -p logs
#RUN cp config  .

CMD ["python3","main.py"]
