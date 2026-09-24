from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import (
    CustomData,
    PredictPipeline
)


application = Flask(__name__)



# HOME PAGE


@application.route("/")
def index():

    return render_template(
        "index.html",
        form={},
        result=None,
        whatifs=None,
        error=None
    )



# CREATE STUDENT DATA


def create_student_data(form):

    return CustomData(

        studytime=float(
            form.get("studytime")
        ),

        absences=float(
            form.get("absences")
        ),

        G1=float(
            form.get("G1")
        ),

        G2=float(
            form.get("G2")
        ),

        age=float(
            form.get("age")
        ),

        failures=float(
            form.get("failures")
        ),

        famsup=form.get("famsup"),

        internet=form.get("internet"),

        sex=form.get("sex"),

        paid=form.get("paid"),

        activities=form.get("activities")

    ).get_data_as_dataframe()



# NORMAL PREDICTION


@application.route(
    "/predict",
    methods=["POST"]
)
def predict_datapoint():

    try:

        student_data = create_student_data(
            request.form
        )

        pipeline = PredictPipeline()

        result = pipeline.predict(
            student_data
        )

        return render_template(

            "index.html",

            form=request.form,

            result=result,

            whatifs=None,

            error=None

        )

    except Exception as e:

        return render_template(

            "index.html",

            form=request.form,

            result=None,

            whatifs=None,

            error=str(e)

        )



# WHAT-IF SIMULATOR


@application.route(
    "/what-if",
    methods=["POST"]
)
def what_if():

    try:

        pipeline = PredictPipeline()

        # Current prediction
        current_data = create_student_data(
            request.form
        )

        current_result = pipeline.predict(
            current_data
        )


        # SCENARIO 1
        # Improve G2 by 2

        scenario1 = request.form.to_dict()

        scenario1["G2"] = str(
            min(
                20,
                float(request.form.get("G2")) + 2
            )
        )

        result1 = pipeline.predict(
            create_student_data(scenario1)
        )


        # SCENARIO 2
        # Reduce absences by 5

        scenario2 = request.form.to_dict()

        scenario2["absences"] = str(
            max(
                0,
                float(
                    request.form.get("absences")
                ) - 5
            )
        )

        result2 = pipeline.predict(
            create_student_data(scenario2)
        )


        # SCENARIO 3
        # Increase study time

        scenario3 = request.form.to_dict()

        scenario3["studytime"] = str(
            min(
                4,
                float(
                    request.form.get("studytime")
                ) + 1
            )
        )

        result3 = pipeline.predict(
            create_student_data(scenario3)
        )


        # SCENARIO 4
        # Improve G2 + reduce absences

        scenario4 = request.form.to_dict()

        scenario4["G2"] = str(
            min(
                20,
                float(
                    request.form.get("G2")
                ) + 2
            )
        )

        scenario4["absences"] = str(
            max(
                0,
                float(
                    request.form.get("absences")
                ) - 5
            )
        )

        result4 = pipeline.predict(
            create_student_data(scenario4)
        )


        whatifs = [

            {
                "name":
                    "Improve G2 by 2 points",

                "result":
                    result1
            },

            {
                "name":
                    "Reduce absences by 5",

                "result":
                    result2
            },

            {
                "name":
                    "Increase study time by 1",

                "result":
                    result3
            },

            {
                "name":
                    "Improve G2 + reduce absences",

                "result":
                    result4
            }

        ]


        return render_template(

            "index.html",

            form=request.form,

            result=current_result,

            whatifs=whatifs,

            error=None

        )


    except Exception as e:

        return render_template(

            "index.html",

            form=request.form,

            result=None,

            whatifs=None,

            error=str(e)

        )



# TEACHER DASHBOARD


@application.route("/dashboard")
def dashboard():

    try:

        import os
        import pandas as pd

        from src.components.data_transformation import (
            DataTransformation
        )

        from src.utils import load_object


        # Load dataset

        data_path = os.path.join(
            "data",
            "student_data.csv"
        )

        df = pd.read_csv(
            data_path
        )


        # Load trained TabPFN

        model = load_object(
            os.path.join(
                "artifacts",
                "model.pkl"
            )
        )

        metadata = load_object(
            os.path.join(
                "artifacts",
                "best_tabpfn_metadata.pkl"
            )
        )


        # Prepare data

        x_df = df.drop(

            columns=[
                "student_id",
                "performance"
            ],

            errors="ignore"

        )


        X = DataTransformation.transform_dataframe(
            x_df
        ).to_numpy(
            dtype="float32"
        )


        # Predict all students

        probabilities = model.predict_proba(
            X
        )[:, 1]


        threshold = float(
            metadata.get(
                "threshold",
                0.50
            )
        )


        predictions = (
            probabilities >= threshold
        ).astype(int)


        # Risk groups

        low = int(
            (probabilities < 0.25).sum()
        )

        moderate = int(
            (
                (probabilities >= 0.25)
                &
                (probabilities < 0.60)
            ).sum()
        )

        high = int(
            (probabilities >= 0.60).sum()
        )


        average_risk = round(
            float(
                probabilities.mean()
            ) * 100,
            1
        )


        # Student table

        students = []


        for i, probability in enumerate(
            probabilities
        ):

            if probability < 0.25:

                level = "LOW"

            elif probability < 0.60:

                level = "MODERATE"

            else:

                level = "HIGH"


            prediction = (

                "At Risk"

                if predictions[i] == 1

                else

                "Good Performance"

            )


            if "student_id" in df.columns:

                student_id = df.iloc[i][
                    "student_id"
                ]

            else:

                student_id = i + 1


            students.append(

                {

                    "student_id":
                        student_id,

                    "prediction":
                        prediction,

                    "risk_score":
                        round(
                            float(
                                probability
                            ) * 100,
                            1
                        ),

                    "level":
                        level

                }

            )


        # Highest risk first

        students.sort(

            key=lambda x:
                x["risk_score"],

            reverse=True

        )


        return render_template(

            "dashboard.html",

            total=len(df),

            low=low,

            moderate=moderate,

            high=high,

            avg=average_risk,

            rows=students[:25],

            error=None

        )


    except Exception as e:

        return render_template(

            "dashboard.html",

            error=str(e)

        )



# RUN FLASK


if __name__ == "__main__":

    application.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )