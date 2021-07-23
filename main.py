import streamlit as st
from fonctions import *
import socket


image=im.open('fav.ico')
st.set_page_config(page_title='Station',page_icon=image,  layout = 'wide', initial_sidebar_state = 'auto')






hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def main():
	

	st.title("Detection WebApp")

	menu = ["Home","Login"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.subheader("Home")
		video_file = open('Streamlit.mp4', 'rb')
		video_bytes = video_file.read()
		st.video(video_bytes)

	elif choice == "Login":

		username = st.sidebar.text_input("User Name")
		password = st.sidebar.text_input("Password",type='password')
		hostname = socket.gethostname()
		ipadd = socket.gethostbyname(hostname)

		if st.sidebar.checkbox("Login"):
			
			create_usertable()
			hashed_pswd = make_hashes(password)


			if username=="admin" and password == "admin":
				result =True
			else :
				result = login_user(username,check_hashes(password,hashed_pswd),ipadd)
			if result:

				menu = ["Dashboard","Station","Shop"]
				choice = st.sidebar.selectbox("Menu",menu)

				if choice == "Dashboard":
					st.subheader('Database , *Visualization!* :sunglasses:')
					# Infos
					show_data()

				elif choice == "Station":
					st.subheader("Station Detection")
					station_detection()
							

				elif choice == "Shop":
					st.subheader("Shop Detection")
					shop_detection()
					

				
			else:
				st.warning("Incorrect Username/Password")






if __name__ == '__main__':
	main()