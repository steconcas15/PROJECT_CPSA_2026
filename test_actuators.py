"""
import sys
import time
from mbientlab.metawear import MetaWear, libmetawear
from mbientlab.metawear.cbindings import *
from ctypes import byref

MAC = "f1:82:64:cb:61:d3"
device = MetaWear(MAC)
device.connect()

print("Test Buzzer: prova suono per 1 secondo...")

# mbl_mw_buzzer_play: (board, pulse_width, repeat_count)
# pulse_width: durata dell'impulso
# repeat_count: numero di ripetizioni
# Il comando richiede la configurazione del buzzer
libmetawear.mbl_mw_buzzer_play(device.board, 1000) 

time.sleep(1)

# Ferma il buzzer
libmetawear.mbl_mw_buzzer_stop(device.board)

device.disconnect()
print("Disconnesso.")
"""

import time
from mbientlab.metawear import MetaWear, libmetawear
from mbientlab.metawear.cbindings import *
from ctypes import byref

# Inserisci il tuo MAC address rilevato
MAC = "f1:82:64:cb:61:d3"
print(f"Connecting to {MAC}...")
device = MetaWear(MAC)
device.connect()
print("Connected! Testing motor...")

pattern = LedPattern(repeat_count=Const.LED_REPEAT_INDEFINITELY)
libmetawear.mbl_mw_led_load_preset_pattern(byref(pattern), LedPreset.SOLID)
libmetawear.mbl_mw_led_write_pattern(device.board, byref(pattern), LedColor.GREEN)

libmetawear.mbl_mw_led_play(device.board)

sleep(5)

libmetawear.mbl_mw_led_stop_and_clear(device.board)

device.disconnect()
print("Disconnected.")

