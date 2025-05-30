import datetime

from art import *


def print_start_message():
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tprint("", decoration="love_music")
    print(f"Starting data preparation at {date}")
    tprint("", decoration="love_music")
    print()
    return date


def print_end_message(date_start):
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print()
    tprint("", decoration="love_music")
    print(f"Finished model training at {date}")
    print(
        "Time taken for data preparation and training: "
        f"{datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(date_start, '%Y-%m-%d %H:%M:%S')}"
    )
    tprint("", decoration="love_music")
    print()


def print_model_summary(model):
    tprint("", decoration="love_music")
    model.summary()
    print()
    tprint("", decoration="love_music")
    print()
