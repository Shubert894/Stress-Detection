import sys
import json
import time
import struct
import numpy as np
import os
from pathlib import Path

import bluetooth
from bluetooth.btcommon import BluetoothError

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import algorithms.processing.primitives as pv

def connect_bluetooth_addr(addr):
    for i in range(5):
        if i > 0:
            time.sleep(1)
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        try:
            sock.connect((addr, 1))
            sock.setblocking(False)
            return sock, addr
        except BluetoothError:
            print("ERROR!")
            raise
    return None

def start_headset(addr):
    socket, socketAddress = connect_bluetooth_addr(addr)
    if socket is None:
        print('Connection failed.')
        sys.exit(-1)


    for i in range(5):
        try:
            if i>0:
                print("Retrying...")
            time.sleep(1)
            len(socket.recv(10))
            break
        except BluetoothError:
            print("BluetoothError")
        except:
            print('...')
        if i == 5:
            print("Conenection failed.")
            sys.exit(-1)

    print(f"Connected to the headset at {socketAddress}")

    return socket

class DataRecorder:
    def __init__(self):
        self.meditation = []
        self.attention = []
        self.raw = []
        self.blink = []
        self.poor_signal = []

        self.attention_queue = []
        self.meditation_queue = []
        self.poor_signal_queue = []
        self.blink_queue = []
        self.raw_queue = []

    def get_last_n_raw_second(self, n):
        if len(self.raw)>= 512*n:
            return self.raw[len(self.raw)-512*n:] 
        else:
            return self.raw
    
    def get_last_n_poor_signal(self, n):
        if len(self.poor_signal)>= 512*n:
            return self.poor_signal[len(self.poor_signal)-512*n:] 
        else:
            return self.poor_signal
    
    def get_last_n_blink(self, n):
        if len(self.blink)>= 512*n:
            return self.blink[len(self.blink)-512*n:] 
        else:
            return self.blink

    def dispatch_data(self, key, value):
        if key == "attention":
            self.attention_queue.append(value)
            # Blink and "poor signal" is only sent when a blink or poor signal is detected
            # So fake continuous signal as zeros.
            
            
        elif key == "meditation":
            self.meditation_queue.append(value)
        elif key == "raw":
            # self.blink_queue.append(0)
            # self.poor_signal_queue.append(0)
            self.raw_queue.append(value)
        elif key == "blink":
            # self.blink_queue.append(value)
            if len(self.blink_queue)>0:
                self.blink_queue[-1] = value
 
        elif key == "poor_signal":
            if len(self.poor_signal_queue)>0:
                self.poor_signal_queue[-1] = value
     
    def record_meditation(self, attention):
        self.meditation_queue.append()
        
    def record_blink(self, attention):
        self.blink_queue.append()
    
    def finish_chunk(self):
        """ called periodically to update the timeseries """
        self.meditation += self.meditation_queue
        self.attention += self.attention_queue
        self.blink += self.blink_queue
        self.raw += self.raw_queue
        self.poor_signal += self.poor_signal_queue

        self.attention_queue = []
        self.meditation_queue = []
        self.poor_signal_queue = []
        self.blink_queue = []
        self.raw_queue = []

class DataParser(object):
    def __init__(self, recorder):
        self.recorder = recorder
        self.parser = self.parse()
        self.parser.__next__()

    def feed(self, data):
        for c in data:
            self.parser.send(ord(chr(c)))
        self.recorder.finish_chunk()
    
    def dispatch_data(self, key, value):
        recorder.dispatch_data(key, value)

    def parse(self):
        """
            This generator parses one byte at a time.
        """
        i = 1
        times = []
        while 1:
            byte = yield
            if byte== 0xaa:
                byte = yield # This byte should be "\aa" too
                if byte== 0xaa:
                    # packet synced by 0xaa 0xaa
                    packet_length = yield
                    packet_code = yield
                    if packet_code == 0xd4:
                        # standing by
                        self.state = "standby"
                    elif packet_code == 0xd0:
                        self.state = "connected"
                    elif packet_code == 0xd2:
                        data_len = yield
                        headset_id = yield
                        headset_id += yield
                        self.dongle_state = "disconnected"
                    else:
                        self.sending_data = True
                        left = packet_length - 2
                        while left>0:
                            if packet_code ==0x80: # raw value
                                row_length = yield
                                a = yield
                                b = yield
                                value = struct.unpack("<h",bytes([b, a]))[0]
                                self.dispatch_data("raw", value)
                                left -= 2
                            elif packet_code == 0x02: # Poor signal
                                a = yield
                                self.dispatch_data("poor_signal", a)
                                left -= 1
                            elif packet_code == 0x04: # Attention (eSense)
                                a = yield
                                if a>0:
                                    v = struct.unpack("b",bytes([a]))[0]
                                    if 0 < v <= 100:
                                        self.dispatch_data("attention", v)
                                left-=1
                            elif packet_code == 0x05: # Meditation (eSense)
                                a = yield
                                if a>0:
                                    v = struct.unpack("b",bytes([a]))[0]
                                    if 0 < v <= 100:
                                        self.dispatch_data("meditation", v)
                                left-=1
                                
                                
                            elif packet_code == 0x16: # Blink Strength
                                self.current_blink_strength = yield
                                self.dispatch_data("blink", self.current_blink_strength)
                                left-=1
                            elif packet_code == 0x83:
                                vlength = yield
                                self.current_vector = []
                                for row in range(8):
                                    a = yield
                                    b = yield
                                    c = yield
                                    value = a*255*255+b*255+c
                                left -= vlength
                                self.dispatch_data("bands", self.current_vector)
                            packet_code = yield
                else:
                    pass # sync failed
            else:
                pass # sync failed

def save_data():
    d = {}

    for i in range(0,len(timePoints)-1,2):
        d[str(i//2 + 1)] = recorder.raw[timePoints[i]:timePoints[i+1]]

    file = os.path.join(Path.cwd(), Path("data/unconventioned/stress_exp_2.json"))
    with open(file, 'w') as f:
        json.dump(d, f)
    
def on_press(event):
    if event.key == ' ':
        timePoints.append(len(recorder.raw))
        print('Time point taken')

def plot_real_time():
    fig, axs = plt.subplots(3)
    lines=[]
    initial = np.arange(512)
    t = np.arange(0,512)

    freq, pwr = pv.get_power(initial, 512)


    l0, = axs[0].plot(t, initial)
    l1, = axs[1].plot(t, initial)
    l2, = axs[2].plot(freq, pwr)

    
    lines.append(l0)
    lines.append(l1)
    lines.append(l2)

    axs[0].set_ylim(-2000,2000)
    axs[1].set_ylim(-5,5)
    axs[2].set_ylim(-1,6)

    def animate(i):
        try:
            byteData = socket.recv(20000)
            parser.feed(byteData)
        except:        
            # print('Failed to receive the bytes')
            pass
        

        if len(recorder.raw) >= 512:
            lastSR = recorder.get_last_n_raw_second(1)
            lastS = pv.standardize(pv.filter_data(lastSR,512))
            f, power = pv.get_power(lastS, 512)
            
            lines[0].set_data(t, lastSR)
            lines[1].set_data(t, lastS)
            lines[2].set_data(f, power)
        
        return lines
       
    fig.canvas.mpl_connect('key_press_event', on_press)
    ani = FuncAnimation(fig, animate, interval = 20, blit = True)
    plt.show()


if __name__ == '__main__':
    '''
        Establishes a bluetooth link with the headset, given an address.

        Displays the incoming signal in real time, together with its standardized version, and its power spectrum.

        When space bar is pressed the data start to be recorded. When the space bar is pressed the second time
        it ends the recording session and saves the collected signal to the 'src/data/unconventioned/' file.
        Multiple recordings are possible within a run.
    '''
    address = '0D:00:18:A3:0C:BD'
    socket = start_headset(address)
    recorder = DataRecorder()
    parser = DataParser(recorder)

    timePoints = []

    plot_real_time()

    print(timePoints)
    save_data()
