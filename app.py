from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

@app.route('/')
def formulario():
    return render_template('predic_model.html')


@app.route('/predict', methods=['POST'])
def predecir():
    # Obtener los datos del formulario
    person_age = float(request.form['person_age'])
    person_income = float(request.form['person_income'])
    person_emp_length = float(request.form['person_emp_length'])
    loan_amnt = float(request.form['loan_amnt'])
    loan_int_rate = float(request.form['loan_int_rate'])
    loan_percent_income = float(request.form['loan_percent_income'])
    cb_person_cred_hist_length = float(request.form['cb_person_cred_hist_length'])

    # Leer valores de las listas desplegables
    p_h_o_trans = float(request.form['p_h_o_trans'])
    loan_intent_trans = float(request.form['loan_intent_trans'])
    loan_grade_trans = float(request.form['loan_grade_trans'])
    cb_pdf_trans = float(request.form['cb_pdf_trans'])

    # Convertir los datos en una matriz de características
    dato_a_predecir = np.array([
        [person_age, person_income, person_emp_length, loan_amnt, loan_int_rate, 
         loan_percent_income, cb_person_cred_hist_length, p_h_o_trans, 
         loan_intent_trans, loan_grade_trans, cb_pdf_trans]
    ])

    model_load = joblib.load('model_credit_dt.plk')

    # Realizar la predicción con el modelo
    prediccion = model_load.predict(dato_a_predecir)

    # Imprimir la predicción en la consola
    print("La predicción es:", prediccion)

    return ''

    # Devolver la predicción como respuesta en formato JSON
    #return jsonify({'prediccion': prediccion.tolist()})
    #return render_template('predic_model.html', prediccion=prediccion.tolist())


if __name__ == '__main__':
    app.run(debug=True)
