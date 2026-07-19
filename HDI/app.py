from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("hdi_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))


def get_result_class(result):
    result_lower = str(result).lower()
    if "very high" in result_lower or "high" in result_lower:
        return "success"
    if "medium" in result_lower:
        return "warning"
    if "low" in result_lower:
        return "danger"
    return "neutral"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/launch")
def launch():
    return render_template("project.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    form_values = {
        "life_expectancy": request.form.get("life_expectancy", ""),
        "mean_years_schooling": request.form.get("mean_years_schooling", ""),
        "expected_years_schooling": request.form.get("expected_years_schooling", ""),
        "gni_per_capita": request.form.get("gni_per_capita", ""),
    }

    if request.method == "GET":
        return render_template("project.html", form_values=form_values)

    try:
        life_expectancy = float(form_values["life_expectancy"])
        mean_years_schooling = float(form_values["mean_years_schooling"])
        expected_years_schooling = float(form_values["expected_years_schooling"])
        gni_per_capita = float(form_values["gni_per_capita"])

        features = np.array([[
            life_expectancy,
            mean_years_schooling,
            expected_years_schooling,
            gni_per_capita
        ]])

        prediction = model.predict(features)
        result = label_encoder.inverse_transform(prediction)[0]

        return render_template(
            "project.html",
            prediction_text=f"Predicted HDI Category: {result}",
            prediction_class=get_result_class(result),
            form_values=form_values,
        )

    except Exception as e:
        return render_template(
            "project.html",
            prediction_text=f"Error: {str(e)}",
            prediction_class="danger",
            form_values=form_values,
        )


if __name__ == "__main__":
    app.run(debug=True)