import socket
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image as im
#Model requirements
import numpy as np
from streamlit_webrtc import (
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
import av
import queue
from typing import List, NamedTuple
import torch
from datetime import date, datetime

## Time 
now = datetime.now()

# webrtc settings

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": True},
)

# Official Model
MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Custom Model
PLATE = torch.hub.load('ultralytics/yolov5', 'custom', path='./plates.pt')
OCR= torch.hub.load('ultralytics/yolov5', 'custom', path='./ocr_best.pt')
GENDER= torch.hub.load('ultralytics/yolov5', 'custom', path='./gender.pt')
# DB Management
import sqlite3 
conn = sqlite3.connect('data.db',check_same_thread = False)
c = conn.cursor()



# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

### History of use
def create_history():
    c.execute('CREATE TABLE IF NOT EXISTS history(host VARCHAR,timeh VARCHAR)')

create_history()

def add_device(host,timeh):
    c.execute('INSERT INTO history(host, timeh) VALUES (?,?)',(host,timeh))
    conn.commit()

hostname = socket.gethostname()
host = socket.gethostbyname(hostname)
timeh= now.strftime("%d/%m/%Y %H:%M:%S")
add_device(host,timeh)


### End of History check

#Users table

def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT,ipadd TEXT)')


def add_userdata(username,password,ipadd):
	c.execute('INSERT INTO userstable(username,password,ipadd) VALUES (?,?,?)',(username,password,ipadd))
	conn.commit()

def login_user(username,password,ipadd):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ? AND ipadd = ?',(username,password,ipadd))
	data = c.fetchall()
	return data

def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data
# Cars table -- Station detection
def create_carstable():
	c.execute('CREATE TABLE IF NOT EXISTS carstable(id INT AUTO_INCREMENT ,type TEXT,licence TEXT,date TEXT,time TEXT)')


def add_car(type,licence,date,time):
	c.execute('INSERT INTO carstable(type,licence,date,time) VALUES (?,?,?,?)',(type,licence,date,time))   
	conn.commit()  #time= now.strftime("%d/%m/%Y %H:%M:%S")

def view_all_cars(date):
	c.execute('SELECT * FROM carstable WHERE date=?',(date,))
	data = c.fetchall()
	return data

def create_vehicules():
    c.execute('CREATE TABLE IF NOT EXISTS vehicules(id INT AUTO_INCREMENT ,type TEXT,licence TEXT UNIQUE,temps INT)')
def vehicules():
   c.execute('INSERT INTO vehicules (type, licence, temps) SELECT type, licence, MAX(CAST(time AS INTEGER))-MIN(CAST(time AS INTEGER)) FROM carstable GROUP BY licence ON CONFLICT(vehicules.licence) DO UPDATE SET temps = (SELECT MAX(CAST(time AS INTEGER))-MIN(CAST(time AS INTEGER)) FROM carstable WHERE vehicules.licence =carstable.licence)')

def show_vehicules():
    c.execute('SELECT * FROM vehicules')
    data = c.fetchall()
    return data

def average_time():
    c.execute('SELECT AVG(temps) FROM vehicules')
    data = c.fetchall()
    return data

# Persons table -- Shop detection
def create_shoptable():
	c.execute('CREATE TABLE IF NOT EXISTS shoptable(id INT AUTO_INCREMENT ,gender TEXT,date,time TEXT)')


def add_person(gender,date,time):
	c.execute('INSERT INTO shoptable(gender,date,time) VALUES (?,?,?)',(gender,date,time))
	conn.commit()

def view_all_persons():
	c.execute('SELECT * FROM shoptable')
	data = c.fetchall()
	return data

# fonctions de détection (streamlit webrtc)

## Station detection
def station_detection():
    
    class Detection(NamedTuple):
        name: str
        prob: float

    class YOLOv5VideoProcessor(VideoProcessorBase):
        result_queue: "queue.Queue[List[Detection]]"

        def __init__(self) -> None:
            global MODEL
            MODEL.classes=[2,3,4,7]
            self._model = MODEL
            self.result_queue = queue.Queue()

            

        def _annotate_image(self, image, results):
            # loop over the detections
            (h, w) = image.shape[:2]
            result: List[Detection] = []
            create_carstable()

            for _, s in results.pandas().xyxy[0].iterrows():
                cartime = datetime.now()
                xmin, ymin, xmax, ymax = s['xmin'],s['ymin'], s['xmax'], s['ymax']
                box = image[int(ymin):int(ymax), int(xmin):int(xmax)]
                imgp = im.fromarray(box)
                plate=PLATE(imgp, size=416)
                licence_plate =[]
                dict={}
                for _, p in plate.pandas().xyxy[0].iterrows() :
                    xmin, ymin, xmax, ymax = p['xmin'],p['ymin'], p['xmax'], p['ymax']
                    pl = box[int(ymin):int(ymax), int(xmin):int(xmax)]
                    imgo =im.fromarray(pl)
                    licence =OCR(imgo, size=416)
                    for _, l in licence.pandas().xyxy[0].iterrows():
                        dict[l['xmin']]=l['name']
                        #licence_plate.append(l['name'])
                
                for key in sorted(dict):
                    licence_plate.append(dict[key])

                #test = im.fromarray(box)
                #test.save('test.jpg')
                lic = ''.join([str(item) for item in licence_plate])
                add_car(s['name'],lic,str(cartime.date()),str(cartime.time()))
                
                result.append(Detection(name=s['name'], prob=s['confidence']))

            results.render()
            image = results.imgs[0]
                
            return image, result

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="rgb24")
            results = self._model(image)
            annotated_image, result = self._annotate_image(image, results)
            # NOTE: This `recv` method is called in another thread,
            # so it must be thread-safe.
            self.result_queue.put(result)

            return av.VideoFrame.from_ndarray(annotated_image, format="rgb24")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=YOLOv5VideoProcessor,
        async_processing=True,
    )

    if st.checkbox("Show the detected labels", value=True):
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            # NOTE: The video transformation with object detection and
            # this loop displaying the result labels are running
            # in different threads asynchronously.
            # Then the rendered video frames and the labels displayed here
            # are not strictly synchronized.
            while True:
                if webrtc_ctx.video_processor:
                    try:
                        result = webrtc_ctx.video_processor.result_queue.get(
                            timeout=1.0
                        )
                    except queue.Empty:
                        result = None
                    labels_placeholder.table(result)
                else:
                    break

## Shop detection
def shop_detection():
    
    class Detection(NamedTuple):
        name: str
        prob: float

    class YOLOv5VideoProcessor(VideoProcessorBase):
        result_queue: "queue.Queue[List[Detection]]"

        def __init__(self) -> None:
            global MODEL
            MODEL.classes=[0]
            self._model = GENDER
            self.result_queue = queue.Queue()

        def _annotate_image(self, image, results):
            # loop over the detections
            (h, w) = image.shape[:2]
            result: List[Detection] = []
            create_shoptable()

            for _, s in results.pandas().xyxy[0].iterrows():
                timep = datetime.now()
                add_person(s['name'],str(timep.date()),str(timep.time()))
                result.append(Detection(name=s['name'], prob=s['confidence']))

            results.render()
            image = results.imgs[0]
                
            return image, result

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="rgb24")
            results = self._model(image)
            annotated_image, result = self._annotate_image(image, results)

            # NOTE: This `recv` method is called in another thread,
            # so it must be thread-safe.
            self.result_queue.put(result)

            return av.VideoFrame.from_ndarray(annotated_image, format="rgb24")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=YOLOv5VideoProcessor,
        async_processing=True,
    )

    if st.checkbox("Show the detected labels", value=True):
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            # NOTE: The video transformation with object detection and
            # this loop displaying the result labels are running
            # in different threads asynchronously.
            # Then the rendered video frames and the labels displayed here
            # are not strictly synchronized.
            while True:
                if webrtc_ctx.video_processor:
                    try:
                        result = webrtc_ctx.video_processor.result_queue.get(
                            timeout=1.0
                        )
                    except queue.Empty:
                        result = None
                    labels_placeholder.table(result)
                else:
                    break

## Dashboard -- Data Visualisation

def show_data():

    #check tables creation
    create_carstable()
    create_shoptable()
    create_vehicules()
    vehicules()

    shop = view_all_persons()
    vehicule= show_vehicules()

    with st.beta_expander("Données de Station"):
        #print avg time
        avt=average_time()[0][0]
        st.write('The average time cars spend :',int(avt*120),'min')
        #### histogramme
        ### time
        st.write('Précisez la date que vous voulez visualiser :')
        start_date = st.slider("Quel jour?",value=date(2021, 7, 10),format="DD-MM-YYYY")
        start=start_date.strftime('%Y-%m-%d')
        station = view_all_cars(start)
        datac = pd.DataFrame(station,columns=["id","type","licence","date","time"])
        datac['time'] = pd.to_datetime(datac['time'])
        hist_values = np.histogram(datac['time'].dt.hour, bins=24, range=(0,24))[0]
        st.bar_chart(hist_values)

        # vehicules type
        st.write('Types de véhicules :')
        datav = pd.DataFrame(vehicule,columns=["id","type","licence","temps"])
        vehicules_df = datav['type'].value_counts().to_frame()
        pv = px.pie(vehicules_df,names=vehicules_df.index, values='type')
        st.plotly_chart(pv,use_container_width=True)

    with st.beta_expander("Données du Shop"):
        datap = pd.DataFrame(shop,columns=["id","gender","date","time"])
        person_df = datap['gender'].value_counts().to_frame()
        #st.dataframe(person_df)
        #person_df = person_df.reset_index()
        p1 = px.pie(person_df,names=person_df.index, values='gender')
        st.plotly_chart(p1,use_container_width=True)

        datag = pd.DataFrame(shop,columns=["id","gender","date","time"])
        datag['time'] = pd.to_datetime(datag['time'])
        hist_values = np.histogram(datag['time'].dt.hour, bins=24, range=(0,24))[0]
        st.bar_chart(hist_values)

    with st.beta_expander("Table de véhicules"):
        datav = pd.DataFrame(vehicule,columns=["id","type","licence","temps"])
        st.table(datav[['type','licence']])

    with st.beta_expander("Table des visiteurs"):
        datagen = pd.DataFrame(shop,columns=["id","gender","date","time"])
        st.table(datagen[['gender','date','time']])
    

    

    


    
