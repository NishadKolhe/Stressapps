from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)

filename = 'stress.pkl'
model = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv-transform.pkl','rb'))



@app.route("/")

def home():
	return render_template("home.html")

@app.route("/submit", methods =["POST"])
def submit():
	#user = request.form["user"]
	#with open("stress.model","rb") as f:
	#	model = pickle.load(f)
	#data =  np.array([user]).reshape(1, -1)
	#print(type(data))
	#res = model.predict([1])
	#return render_template("home.html", msg=res)
	
	if request.method == 'POST':
		msg = request.form["msg"]
		data = [msg]
		vect = cv.transform(data).toarray()
		res = model.predict(vect)
		if res == 1:
			ans1 = "You are having Stress Just Take Rest."
		else:
			ans1 = "You don't have any Stress Enjoy Your Day."
		return render_template("home.html", ans=ans1)

if __name__ == "__main__":
	app.run(debug=True)		

