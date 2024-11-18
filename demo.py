import os
from pathlib import Path
import pandas as pd
import numpy as np
import dill
from PIL import Image
import streamlit as st
from catboost import CatBoostClassifier

from eclyon.transforms import process_df


path_to_repo = Path(__file__).parent.resolve()
path_to_data = path_to_repo / 'data' / 'clean_health.csv'



def display_drinker_status(index):
    # compute model prediction
    pred = st.session_state.model.predict([st.session_state.X.values[index]])[0]
    pred  = int(pred )
    true = int(st.session_state.y[index])

    # display actual and predicted prices
    col_pred, col_gold = st.columns(2)
    with col_pred:
        st.subheader('predicted status')
        if bool(pred) == True: 
            st.write("Drinker")
        else: 
            st.write("Non-drinker")
    with col_gold:
        st.subheader('real status')
        if bool(true) == True: 
            st.write("Drinker")
        else: 
            st.write("Non-drinker")
    return


def display_features(index):
    st.subheader('Individual features')
    feat0, val0, feat1, val1 = st.columns([3.5, 1.5, 3.5, 1.5])
    row = st.session_state.X.values[index]
    for i, feature in enumerate(st.session_state.X.columns):
        ind = i % 2
        if ind == 0:
            with feat0:
                st.info(feature)
            with val0:
                st.success(str(row[i]))
        elif ind == 1:
            with feat1:
                st.info(feature)
            with val1:
                st.success(str(row[i]))
    return


def init_session_state():
    # session state
    if 'loaded' not in st.session_state:
        # validation set given in notebook
        n_valid = 10000

        # import raw data
        df_raw = pd.read_csv(path_to_data)

        # preprocess data
        X, y, nas = process_df(df_raw, 'drinker')
        X, y = X[n_valid:], y[n_valid:]

        # load regression model
        path_to_model = path_to_repo / 'models'
        model = CatBoostClassifier().load_model(fname = (path_to_model / 'model_full').resolve())

       
        # store in cache
        st.session_state.loaded = True
        st.session_state.n_valid = n_valid
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.model = model
    


def app():
    init_session_state()
    st.title('Choose an individual')
    options = ['-'] + list(range(1, st.session_state.n_valid + 1))
    index = st.selectbox(label = 'Choose an index', options = options, index = 0)
    if index != '-':
        display_drinker_status(index)
        display_features(index)
    return


if __name__ == '__main__':
    app()