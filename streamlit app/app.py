import pandas as pd
import numpy as np
import joblib
import streamlit as st

st.set_page_config("Machine failure prediction app",layout='wide')
st.title("Machine Failure and Failure type prediction system")

##model loading
rfc1= joblib.load(r"C:\Users\prana\Downloads\M6\stage1_bestrf_model.pkl")
rfc2 = joblib.load(r"C:\Users\prana\Downloads\M6\stage2_bestrf_model.pkl")
scaler = joblib.load(r"C:\Users\prana\Downloads\M6\scaler.pkl")

##Sidebar
st.sidebar.header("Available Options")
mode = st.sidebar.radio("Choose any one",['Single prediction','Batch prediction'])

###Features
features = ['Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]',
        'temp_diff(k)',
        'power[kw]'
        ]
cat_feature = 'Type'

total_features = features + [cat_feature]


###single prediction
if mode =="Single prediction":
    st.title("Enter the inputs")
    input_df={}
    col1,col2 = st.columns(2)
    for i,f in enumerate(features):
        if i%2==0:
            with col1:
                input_df[f] = st.number_input(f,value=0.0)
        else:
            with col2:
                input_df[f] = st.number_input(f,value=0.0)
    
    ##categorical input
    input_df[cat_feature] = st.selectbox('Type',[0,1,2])
    input_df = pd.DataFrame([input_df])
    input_df1=input_df.copy()
    input_df.columns = (
            input_df.columns
            .str.replace('[','', regex=False)
            .str.replace(']','', regex=False)
            .str.replace('<','',regex=False)
        )
    st.dataframe(input_df)

    ##Prediction
    if st.button('predict'):
        ##stage1 prediction
        ##scaling numeric features
        features1 =['Air temperature K',
        'Process temperature K',
        'Rotational speed rpm',
        'Torque Nm',
        'Tool wear min',
        'temp_diff(k)',
        'powerkw'
        ]
        input_num_scaled= scaler.fit_transform(input_df[features1])
        input_scal_df = pd.DataFrame(input_num_scaled,columns=features1)

        ##adding categorical column
        input_scal_df[cat_feature] = input_df[cat_feature].values
        ##prediction
        stage1_pred = rfc1.predict(input_scal_df)[0]
        st.subheader("STAGE1 RESULT")
        st.success(f"Machine failure prediction: {stage1_pred}")

        ####stage2 prediction
        if stage1_pred ==1:
            reo1=input_df1.pop('Type')
            input_df1.insert(0,'Type',reo1)
            stage2_pred = rfc2.predict(input_df1)[0]
            st.subheader("STAGE2 PREDICTION")
            st.success(f"Failure type prediction: {stage2_pred}")
        else:
            st.success("Machine is working well, no need for stage2 prediction")

####BATCH PREDICTION
else:
    st.subheader("Upload csv file")
    file = st.file_uploader("Upload a csv",type=['csv'])

    if file is not None:
        st.write("FILE UPLOAED SUCCESSFULLY")
        data = pd.read_csv(file)
        st.dataframe(data.head())

        ##feature engineering
        data['temp_diff(k)'] = data['Process temperature [K]'] - data['Air temperature [K]']
        data['power[kw]'] = round((data['Torque [Nm]']* data['Rotational speed [rpm]'])/9548.8,2)

        ##features
        x=data[total_features]
    
        ##label encoding
        x['Type'] = x['Type'].map({'L':0,'M':1,'H':2})

        ##copy
        x1 = x.copy()

        ###stage1 prediction for batch
        if st.button('predict'):
            x.columns=(
                x.columns
                .str.replace('[','', regex=False)
                .str.replace(']','', regex=False)
                .str.replace('<','',regex=False)
                )
            x_num_scal = scaler.fit_transform(x.drop(['Type'],axis=1))
            x_scal_df = pd.DataFrame(x_num_scal,columns=x.columns[:-1])

            ##add cat column
            x_scal_df[cat_feature] = x[cat_feature].values
            ##prediction
            stage1_prediction = rfc1.predict(x_scal_df)
            data['stage1_prediction'] = stage1_prediction
            st.success('Machine failure detection sucessfull')
            st.subheader("stage1 prediction results")
            st.dataframe(data.head())

            stage1_rows = data[data['stage1_prediction']==1]

            if len(stage1_rows)>0:
                ##rearranging columns
                cols = list(x1.columns) + [stage1_rows.columns[-1]]
                x_stage2 = stage1_rows[cols]
                
                reo1=x_stage2.pop('Type')
                x_stage2.insert(0,'Type',reo1)

                ##label encoding
                x_stage2['Type'] = x_stage2['Type'].map({'L':0,'M':1,'H':2})
                
                ##prediction
                stage2_prediction = rfc2.predict(x_stage2.drop(['stage1_prediction'],axis=1))
                x_stage2['stage2_prediction'] = stage2_prediction

                st.success("Failure type prediction successfull")
                st.subheader("STAGE2 PREDICTION")

                ##mapping target variable to character values
                x_stage2['stage2_prediction'] = x_stage2['stage2_prediction'].map({1:'TWF',2:'OSF',3:'HDF',4:'PWF'})
                st.dataframe(x_stage2.head())
            else:
                st.success("Machine is working well, no need for stage2 Batch prediction")








st.markdown("--------")
st.sidebar.subheader("MODEL INFORMATION")
st.write("Stage1: Machine Failure detection")
st.write("Stage2: Failure type detection")