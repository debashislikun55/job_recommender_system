{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "a2fdb47d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Topic  0\n",
      "sale, work, experi, manag, design, data, project, custom, market, product\n",
      "\n",
      "Topic  1\n",
      "design, data, ux, user, experi, ui, visual, product, interact, prototyp\n",
      "\n",
      "Topic  2\n",
      "design, ux, user, sale, ui, interact, custom, prototyp, product, insur\n",
      "\n",
      "Topic  3\n",
      "project, construct, manag, client, subcontractor, owner, bid, budget, cost, scope\n",
      "\n",
      "Topic  4\n",
      "client, consult, market, strategi, growth, firm, strateg, digit, clients, valu\n",
      "\n",
      "Topic  5\n",
      "market, campaign, content, email, media, brand, channel, social, advertis, execut\n",
      "\n",
      "Topic  6\n",
      "sale, construct, project, custom, roostifi, solar, help, roof, lend, understand\n",
      "\n",
      "Topic  7\n",
      "sale, clinic, consult, trial, diagnost, molecular, project, research, biolog, healthcar\n",
      "\n",
      "Topic  8\n",
      "product, roostifi, custom, clinic, account, lend, manag, trial, coursera, client\n",
      "\n",
      "Topic  9\n",
      "diagnost, biolog, molecular, model, clinic, scientif, adivo, healthcar, consult, figur\n",
      "\n",
      "Topic  10\n",
      "client, estimates, birdey, bid, skill, sale, team, solar, data, biolog\n",
      "\n",
      "Topic  11\n",
      "coursera, product, assist, skill, estimates, consult, learn, wish, bid, strong\n",
      "\n",
      "Topic  12\n",
      "roostifi, adivo, consult, healthcar, lend, estimates, bid, statu, assist, insur\n",
      "\n",
      "Topic  13\n",
      "coursera, construct, paypal, diagnost, client, learn, machin, molecular, biolog, statu\n",
      "\n",
      "Topic  14\n",
      "custom, wish, model, commerc, roof, birdey, client, increas, adivo, superintendent\n",
      "\n",
      "Topic  15\n",
      "wish, commerc, account, transact, molecular, paypal, biolog, diagnost, construct, ori\n",
      "\n",
      "Topic  16\n",
      "dropbox, roof, research, estimates, order, data, client, file, assist, awt\n",
      "\n",
      "Topic  17\n",
      "birdey, consult, construct, autodesk, design, dashboard, analyt, yahoo, metric, facebook\n",
      "\n",
      "Topic  18\n",
      "roostifi, coursera, lend, figur, servic, model, solar, data, design, commiss\n",
      "\n",
      "Topic  19\n",
      "dropbox, client, coursera, peopl, wish, survey, internet, research, employ, roostifi\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 432x288 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 720x720 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import process_data as pda\n",
    "import pandas as pd\n",
    "import pca_chart as pc\n",
    "import matplotlib.pyplot as plt\n",
    "import word_similarity\n",
    "import pickle\n",
    "import re\n",
    "\n",
    "#Introduce App\n",
    "st.title('Job Recommender')\n",
    "st.markdown('(Non-Technical Business Roles in 60 - 120k Salary Range + Data Scientists)')\n",
    "st.sidebar.markdown(\"See which jobs best match your profile and optimize your resume / LinkedIn!\")\n",
    "st.sidebar.markdown(\"This app has 3 functionalities:\")\n",
    "st.sidebar.markdown(\"1. Predict which job type you match most with based on your resume / LinkedIn.\")\n",
    "\n",
    "st.sidebar.markdown(\"2. Show which job cluster your resume fits within.\")\n",
    "\n",
    "st.sidebar.markdown(\"3. Help you find which keywords you're missing and matching for your dream job!\")\n",
    "\n",
    "st.sidebar.markdown(\"Scroll Down to See All Functionalities!\")\n",
    "\n",
    "#Get and transform user's resume or linkedin\n",
    "user_input = st.text_area(\"copy or paste your resume or linkdn here\", '')\n",
    "\n",
    "user_input = str(user_input)\n",
    "user_input = re.sub('[^a-zA-Z0-9\\.]', ' ', user_input)\n",
    "user_input = user_input.lower()\n",
    "\n",
    "user_input = pd.Series(user_input)\n",
    "\n",
    "#load NLP + classification models\n",
    "\n",
    "topic_model = pickle.load(open('topic_model.sav', 'rb'))\n",
    "classifier = pickle.load(open('classification_model.sav', 'rb'))\n",
    "vec = pickle.load(open('job_vec.sav', 'rb'))\n",
    "\n",
    "classes, prob = pda.main(user_input, topic_model, classifier, vec)\n",
    "\n",
    "data = pd.DataFrame(zip(classes.T, prob.T), columns = ['jobs', 'probability'])\n",
    "\n",
    "#Plot probability of person belonging to a job class\n",
    "def plot_user_probability():\n",
    "    #plt.figure(figsize = (2.5,2.5))\n",
    "    plt.barh(data['jobs'], data['probability'], color = 'r')\n",
    "    plt.title('Percent Match of Job Type')\n",
    "    st.pyplot()\n",
    "\n",
    "#Plot where user fits in with other job clusters\n",
    "def plot_clusters():\n",
    "    st.markdown('This chart uses PCA to show you where you fit among the different job archetypes.')\n",
    "    X_train, pca_train, y_train, y_vals, pca_model = pc.create_clusters()\n",
    "    for i, val in enumerate(y_train.unique()):\n",
    "        y_train = y_train.apply(lambda x: i if x == val else x)\n",
    "    example = user_input\n",
    "    doc = pc.transform_user_resume(pca_model, example)\n",
    "\n",
    "    pc.plot_PCA_2D(pca_train, y_train, y_vals, doc)\n",
    "    st.pyplot()\n",
    "\n",
    "plot_user_probability()\n",
    "st.title('Representation Among Job Types')\n",
    "plot_clusters()\n",
    "\n",
    "st.title('Find Matching Keywords')\n",
    "st.markdown('This function shows you which keywords your resume either contains or doesnt contain, according to the most significant words in each job description.')\n",
    "st.markdown(\"The displayed keywords are stemmed, ie 'analysis' --> 'analys' and 'commision' --> 'commiss'\")\n",
    "option = st.selectbox(\n",
    "    'Which job would you like to compare to?',\n",
    " ('ux,designer', 'data,analyst', 'project,manager', 'product,manager', 'account,manager', 'consultant', 'marketing', 'sales',\n",
    " 'data,scientist'))\n",
    "\n",
    "st.write('You selected:', option)\n",
    "matches, misses = word_similarity.resume_reader(user_input, option)\n",
    "match_string = ' '.join(matches)\n",
    "misses_string = ' '.join(misses)\n",
    "\n",
    "st.markdown('Matching Words:')\n",
    "st.markdown(match_string)\n",
    "st.markdown('Missing Words:')\n",
    "st.markdown(misses_string)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "00717ad8",
   "metadata": {},
   "outputs": [],
   "source": [
    "from warnings import warn "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "7a578c4a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import warnings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "0cc09415",
   "metadata": {},
   "outputs": [],
   "source": [
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a560fa64",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
