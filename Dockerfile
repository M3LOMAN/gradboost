FROM python

RUN python -m pip install matplotlib tqdm sklearn

WORKDIR /C:/Users/Владимир/OneDrive/Документы/dectree

COPY . .

ENTRYPOINT ["python"]

CMD ["gradboost.py"]