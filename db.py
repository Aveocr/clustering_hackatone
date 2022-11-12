from datetime import datetime

from peewee import *

db = SqliteDatabase('app.db')
db.connect()

class IpData(Model):
    inn = CharField(max_length=12, primary_key=True)
    inn_start_date = DateTimeField()
    inn_end_date = DateTimeField(null=True)
    gender = IntegerField(choices=[(1, 'Male'), (2, 'Female')])
    business_count = IntegerField(default=1)

    class Meta:
        database = db
        table_name = 'ip_data'

db.create_tables([IpData])


def get_ip_data(inn: str) -> IpData | None:
    return IpData.get_or_none(IpData.inn == inn)


def add_ip_data(
    inn: str,
    inn_start_date: datetime,
    inn_end_date: datetime,
    gender: int
    ):
    ip_data = get_ip_data(inn)
    if ip_data is not None:
        if inn_start_date > ip_data.inn_start_date:
            ip_data.inn_start_date = inn_start_date
            ip_data.inn_end_date = inn_end_date
        ip_data.business_count += 1
        ip_data.save()
        return

    IpData.create(
        inn=inn,
        inn_start_date=inn_start_date,
        inn_end_date=inn_end_date,
        gender=gender
    )
