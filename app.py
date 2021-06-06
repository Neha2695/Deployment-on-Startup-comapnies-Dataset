#!/usr/bin/env python
# coding: utf-8

# In[1]:


import flask as f
import pickle as pkl
import pandas as pd


# In[2]:


app = f.Flask(__name__)
model = pkl.load(open("model.pkl","rb"))


# In[3]:


@app.route("/")
def home():
    return f.render_template("Index.html")


# In[4]:


@app.route("/predict",methods=["POST"])
def predict():
    A=[]
    for i in f.request.form.values():
        A.append(int(i))
    pred_prof = model.predict(pd.DataFrame([A[0],A[1]]).T)
    return f.render_template("Index.html",pred="Predicted_Profit: %.2f"%pred_prof)


# In[5]:


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:





# In[ ]:




