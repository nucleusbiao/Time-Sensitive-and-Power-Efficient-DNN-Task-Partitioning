import struct
import pickle
import socket
import time

def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def decode_msg(msg):
    res = pickle.loads(msg)
    #import torch
    #res=torch.load(msg, map_location=torch.device('cpu'))
    return res

def encode_msg(data):
    msg = pickle.dumps(data)
    return msg

class clientCommunication():
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    def connect(self):
        self.s.connect((self.host, self.port))  #w

    def send_msg(self, msg):
        msg = encode_msg(msg)
        encode_time=time.time()
        #self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.s.connect((self.host, self.port))
        send_time=time.time()
        send_msg(self.s, msg)
        return send_time,encode_time
    def receive_msg(self):
        received = recv_msg(self.s)
        recceive_time=time.time()  #w
        received = decode_msg(received)
        return received,recceive_time  #w

    def close_channel(self):
        self.s.close()

class serverCommunication():
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((self.host, self.port))
        self.s.listen()

    def send_msg(self, conn, msg):
        msg = encode_msg(msg)
        send_msg(conn, msg)

    def receive_msg(self, conn):
        received = recv_msg(conn)
        received = decode_msg(received)
        return received

    def accept_conn(self):
        conn, addr = self.s.accept()
        return conn, addr

    def close_channel(self):
        self.s.close()
