import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dateutil import relativedelta


def show_columns_stats(df, columns=None):
    qty_rows = df.shape[0]

    if columns is None:
        columns = df.columns

    offset = 35
    try:
        offset = columns.str.len().max()
    except Exception:
        offset = len(max(columns, key=len))

    for col in columns:
        q = df.count()[col]
        perc = np.round((q / qty_rows * 100), 2)
        missing_perc = np.round((100 - perc), 2)
        missing_qty = df[df[col].isnull()].shape[0]
        uniques = df[col].unique().size
        col_print = (col + ("." * offset))[0:offset]
        print(
            f"{col_print:>} = {q:>5} rows ({perc:>5}%) {missing_qty:>5} with NaN ({missing_perc:>5}%) Uniques= {uniques:>5} "
        )


def date_diff(begin_date, end_date):
    delta = relativedelta.relativedelta(end_date, begin_date)
    return np.abs(delta.years), np.abs(delta.months), np.abs(delta.days)


def calculate_age_from_birth_delta(days):
    today = datetime.now()
    birth = timedelta(days=days)
    diff = today - birth
    years, _, days = date_diff(today, diff)
    return np.abs(years)


def preprocess_als_history(df):

    df_als_history = df.copy()

    df_als_history["site_onset"] = np.NaN
    df_als_history["site_onset"] = df_als_history["site_onset"].astype(str)
    df_als_history.head(3)

    columns = [
        "Site_of_Onset___Bulbar",
        "Site_of_Onset___Limb",
        "Site_of_Onset___Limb_and_Bulbar",
        "Site_of_Onset___Other",
        "Site_of_Onset___Other_Specify",
        "Site_of_Onset___Spine",
    ]

    for col in columns:
        df_als_history.loc[(df_als_history[col] != 1), col] = np.NaN

    df_als_history.loc[
        (df_als_history["Site_of_Onset"] == "Onset: Bulbar"), "site_onset"
    ] = "Bulbar"

    df_als_history.loc[
        (df_als_history["Site_of_Onset___Bulbar"] == 1)
        & (df_als_history["Site_of_Onset"].isnull())
        & (df_als_history["Site_of_Onset___Limb"].isnull())
        & (df_als_history["Site_of_Onset___Spine"].isnull())
        & (df_als_history["Site_of_Onset___Limb_and_Bulbar"].isnull())
        & (df_als_history["Site_of_Onset___Other"].isnull())
        & (df_als_history["Site_of_Onset___Other_Specify"].isnull()),
        "site_onset",
    ] = "Bulbar"

    df_als_history.loc[
        (df_als_history["Site_of_Onset"] == "Onset: Limb")
        | (df_als_history["Site_of_Onset"] == "Onset: Spine"),
        "site_onset",
    ] = "Limb/Spinal"

    df_als_history.loc[
        (df_als_history["Site_of_Onset___Limb"] == 1)
        & (df_als_history["Site_of_Onset___Bulbar"].isnull())
        & (df_als_history["Site_of_Onset"].isnull())
        & (df_als_history["Site_of_Onset___Spine"].isnull())
        & (df_als_history["Site_of_Onset___Limb_and_Bulbar"].isnull())
        & (df_als_history["Site_of_Onset___Other"].isnull())
        & (df_als_history["Site_of_Onset___Other_Specify"].isnull()),
        "site_onset",
    ] = "Limb/Spinal"

    df_als_history.loc[
        (df_als_history["Site_of_Onset___Spine"] == 1)
        & (df_als_history["Site_of_Onset___Limb"].isnull())
        & (df_als_history["Site_of_Onset___Bulbar"].isnull())
        & (df_als_history["Site_of_Onset"].isnull())
        & (df_als_history["Site_of_Onset___Limb_and_Bulbar"].isnull())
        & (df_als_history["Site_of_Onset___Other"].isnull())
        & (df_als_history["Site_of_Onset___Other_Specify"].isnull()),
        "site_onset",
    ] = "Limb/Spinal"

    df_als_history.loc[
        (df_als_history["Site_of_Onset"] == "Onset: Limb and Bulbar"), "site_onset"
    ] = "Bulbar and Limb/Spinal"

    df_als_history.loc[
        (df_als_history["Site_of_Onset___Limb_and_Bulbar"] == 1)
        & (df_als_history["Site_of_Onset___Limb"].isnull())
        & (df_als_history["Site_of_Onset___Bulbar"].isnull())
        & (df_als_history["Site_of_Onset"].isnull())
        & (df_als_history["Site_of_Onset___Spine"].isnull())
        & (df_als_history["Site_of_Onset___Other"].isnull())
        & (df_als_history["Site_of_Onset___Other_Specify"].isnull()),
        "site_onset",
    ] = "Onset: Limb and Bulbar"

    df_als_history.loc[
        (df_als_history["Site_of_Onset"] == "Onset: Other"), "site_onset"
    ] = "Other"

    df_als_history.loc[
        (df_als_history["Site_of_Onset___Other"] == 1)
        & (df_als_history["Site_of_Onset___Other_Specify"].isnull())
        & (df_als_history["Site_of_Onset___Limb_and_Bulbar"].isnull())
        & (df_als_history["Site_of_Onset___Limb"].isnull())
        & (df_als_history["Site_of_Onset___Bulbar"].isnull())
        & (df_als_history["Site_of_Onset"].isnull())
        & (df_als_history["Site_of_Onset___Spine"].isnull()),
        "site_onset",
    ] = "Other"

    df_als_history.loc[
        (df_als_history["Site_of_Onset___Other_Specify"] == 1)
        & (df_als_history["Site_of_Onset___Other"].isnull())
        & (df_als_history["Site_of_Onset___Limb_and_Bulbar"].isnull())
        & (df_als_history["Site_of_Onset___Limb"].isnull())
        & (df_als_history["Site_of_Onset___Bulbar"].isnull())
        & (df_als_history["Site_of_Onset"].isnull())
        & (df_als_history["Site_of_Onset___Spine"].isnull()),
        "site_onset",
    ] = "Other"

    irrelevant_cols = [
        "Site_of_Onset___Bulbar",
        "Site_of_Onset___Limb",
        "Site_of_Onset___Limb_and_Bulbar",
        "Site_of_Onset___Other",
        "Site_of_Onset___Other_Specify",
        "Site_of_Onset___Spine",
        "Disease_Duration",
        "Symptom",
        "Symptom_Other_Specify",
        "Location",
        "Location_Other_Specify",
        "Site_of_Onset",
        "Subject_ALS_History_Delta",
    ]

    df_als_history.drop(
        columns=irrelevant_cols,
        inplace=True,
    )

    df_als_history.rename(
        columns={"Onset_Delta": "Symptoms_Onset_Delta", "site_onset": "Site_Onset"},
        inplace=True,
    )

    return df_als_history


def date_diff_from_days(days_orig, return_abs_value=True):
    begin_date = datetime.fromisoformat("1900-01-01")
    end_date = begin_date + timedelta(days=abs(days_orig))
    years, months, days = date_diff(begin_date=begin_date, end_date=end_date)

    if return_abs_value:
        return np.abs(years), np.abs(months), np.abs(days)
    else:
        if days_orig < 0:
            years = -years
            months = -months
            days = -days

        return years, months, days


def join_datasets_by_key(
    df_main,
    df_to_join,
    key_name,
    how="left",
    lsuffix="",
    rsuffix="",
    sort=False,
    raise_error=False,
):
    rows_df_main = df_main.shape[0]
    df_return = df_main.join(
        df_to_join.set_index(key_name),
        on=key_name,
        how=how,
        lsuffix=lsuffix,
        rsuffix=rsuffix,
        sort=sort,
    )
    rows_return = df_return.shape[0]
    if (how == "left") & (rows_return > rows_df_main):
        msg = "DF_TO_JOIN has duplicated values in KEY column. Remove duplicate keys before joining."
        if raise_error:
            raise NameError(msg)
        else:
            print(f"ERROR: {msg}")
    #
    return df_return


def calculate_months_from_days(days, return_abs_value=True):
    months_total = np.NaN
    if (days is not None) and (not math.isnan(days)):
        years, months, days = date_diff_from_days(
            days, return_abs_value=return_abs_value
        )
        months_total = (years * 12) + months
    if return_abs_value:
        return np.abs(months_total)
    else:
        return months_total


def preprocess_diagnosis_delay(df):
    df["Diagnostic_Delay_in_Days"] = np.abs(df.Symptoms_Onset_Delta) - np.abs(
        df.Diagnosis_Delta
    )
    diagnosis_delay_in_months = df["Diagnostic_Delay_in_Days"].apply(
        lambda x: calculate_months_from_days(x)
    )
    df.loc[df.index, "Diagnostic_Delay"] = diagnosis_delay_in_months
    irrelevant_cols = ["Diagnostic_Delay_in_Days"]
    df.drop(columns=irrelevant_cols, inplace=True)
    return df


def calculate_years_from_days(days):
    years_total = np.NaN
    if (days is not None) and (not math.isnan(days)):
        years, months, days = date_diff_from_days(days)
        years_total = years

        if months in [3, 4]:
            years_total += 0.25
        elif months in [5, 6, 7, 8]:
            years_total = years_total + 0.5
        elif months in [9, 10]:
            years_total += 0.75
        elif months >= 11:
            years_total = years_total + 1.0

    return np.abs(years_total)


def calculate_age_from_onset_delta(age_at_trial, symptoms_onset_delta):
    age_to_subtract = calculate_years_from_days(np.abs(symptoms_onset_delta))
    age_to_subtract = int(age_to_subtract)

    age_at_onset = age_at_trial - age_to_subtract

    if age_to_subtract < 1:
        age_at_onset = np.round(age_at_onset, 0)

    return np.trunc(age_at_onset)


def preprocess_age_at_onset(df_to_process):
    df = df_to_process.copy()
    df["Age_at_Onset"] = np.NaN
    df_calc_age_onset = df.loc[
        (df.Age.notnull()) & (df.Symptoms_Onset_Delta.notnull())
    ].copy()
    ages_calculated = df_calc_age_onset.apply(
        lambda x: calculate_age_from_onset_delta(x["Age"], x["Symptoms_Onset_Delta"]),
        axis=1,
    )
    # update samples with the calculated Age_at_Onset
    df.loc[df_calc_age_onset.index, "Age_at_Onset"] = ages_calculated

    return df


def remove_rows(df, to_delete, info="", verbose=True):
    rows_previous = df.shape[0]
    rows_to_delete = to_delete.shape[0]
    df = df.drop(to_delete.index)
    rows_after = df.shape[0]
    if verbose:
        print(
            f"  - {info} Previous={rows_previous}, To delete={rows_to_delete}, After={rows_after}"
        )
    return df


def preprocess_last_visit(df_to_process, data_dir):
    df = df_to_process.copy()

    data_files = [
        ["PROACT_ALSFRS", "ALSFRS_Delta"],
        ["PROACT_FVC", "Forced_Vital_Capacity_Delta"],
        ["PROACT_DEATHDATA", "Death_Days"],
        ["PROACT_LABS", "Laboratory_Delta"],
        ["PROACT_RILUZOLE", "Riluzole_use_Delta"],
        ["PROACT_SVC", "Slow_vital_Capacity_Delta"],
        ["PROACT_VITALSIGNS", "Vital_Signs_Delta"],
        ["PROACT_ALSHISTORY", "Subject_ALS_History_Delta"],
        ["PROACT_DEMOGRAPHICS", "Demographics_Delta"],
        ["PROACT_ELESCORIAL", "delta_days"],
        ["PROACT_FAMILYHISTORY", "Family_History_Delta"],
        ["PROACT_HANDGRIPSTRENGTH", "MS_Delta"],
        ["PROACT_MUSCLESTRENGTH", "MS_Delta"],
        ["PROACT_TREATMENT", "Treatment_Group_Delta"],
        ["PROACT_ADVERSEEVENTS", "Start_Date_Delta"],
        ["PROACT_ADVERSEEVENTS", "End_Date_Delta"],
        ["PROACT_CONMEDS", "Start_Delta"],
        ["PROACT_CONMEDS", "Stop_Delta"],
    ]

    df_last_visit = pd.DataFrame(data=[], columns=["subject_id", "Delta", "Biomarker"])

    for data_file, col_delta in data_files:
        print(f" - Get Last_Visit registered in {data_file}")
        csv_file = f"{data_dir}/{data_file}.csv"
        df_raw = pd.read_csv(csv_file, delimiter=",", usecols=["subject_id", col_delta])
        df_raw.rename(columns={col_delta: "Last_Visit_Delta"}, inplace=True)
        df_raw.sort_values(["subject_id", "Last_Visit_Delta"])
        df_grouped = df_raw.groupby("subject_id").max()
        df_grouped.reset_index(inplace=True)
        df_grouped["Biomarker"] = data_file

        if df_last_visit.shape[0] == 0:
            df_last_visit = df_grouped.copy()
        else:
            df_last_visit = pd.concat([df_last_visit, df_grouped], ignore_index=True)

    df_last_visit.dropna(inplace=True)
    df_last_visit.sort_values(["subject_id", "Last_Visit_Delta"])
    df_last_visit = df_last_visit.groupby("subject_id").max()
    df_last_visit.reset_index(inplace=True)
    df_to_join = df_last_visit[["subject_id", "Last_Visit_Delta"]].copy()
    df = join_datasets_by_key(
        df_main=df, df_to_join=df_to_join, key_name="subject_id", how="left"
    )
    last_visit_in_months = df.Last_Visit_Delta.apply(
        lambda x: calculate_months_from_days(x)
    )
    df.loc[df.index, "Last_Visit_from_First_Visit"] = last_visit_in_months
    df["Last_Visit_from_Onset_in_Days"] = np.abs(df.Last_Visit_Delta) + np.abs(
        df.Symptoms_Onset_Delta
    )
    last_visit_in_months = df["Last_Visit_from_Onset_in_Days"].apply(
        lambda x: calculate_months_from_days(x)
    )
    df.loc[df.index, "Last_Visit_from_Onset"] = last_visit_in_months
    irrelevant_cols = [
        "Last_Visit_from_Onset_in_Days",
    ]
    df.drop(
        columns=irrelevant_cols,
        inplace=True,
    )
    return df


def preprocess_death_data(df_to_process, data_dir):
    df = df_to_process.copy()
    data_file = f"{data_dir}/PROACT_DEATHDATA.csv"
    df_raw = pd.read_csv(data_file, delimiter=",")
    df_remove_duplicated = df_raw.sort_values(
        ["subject_id", "Subject_Died", "Death_Days"]
    )
    df_remove_duplicated = (
        df_remove_duplicated.groupby(["subject_id", "Subject_Died"])
        .last()
        .reset_index()
    )
    df_raw = df_remove_duplicated.copy()
    df = join_datasets_by_key(
        df_main=df,
        df_to_join=df_raw,
        key_name="subject_id",
        how="left",
        raise_error=True,
    )
    df.rename(
        columns={"Subject_Died": "Event_Dead", "Death_Days": "Event_Dead_Delta"},
        inplace=True,
    )
    df.Event_Dead = df.Event_Dead.map({"Yes": True, "No": False, np.NaN: False})
    df_event_dead_false = df.loc[(df.Event_Dead == False)].copy()
    df_event_dead_false["Event_Dead_Time_from_Onset"] = df_event_dead_false[
        "Last_Visit_from_Onset"
    ]
    df.loc[df_event_dead_false.index, "Event_Dead_Time_from_Onset"] = (
        df_event_dead_false["Event_Dead_Time_from_Onset"]
    )
    df_event_dead_false["Event_Dead_Time_from_First_Visit"] = df_event_dead_false[
        "Last_Visit_from_First_Visit"
    ]
    df.loc[df_event_dead_false.index, "Event_Dead_Time_from_First_Visit"] = (
        df_event_dead_false["Event_Dead_Time_from_First_Visit"]
    )
    df_event_dead_true = df.loc[
        (df.Event_Dead == True) & (df.Event_Dead_Delta.isnull())
    ].copy()
    df.loc[df_event_dead_true.index, "Event_Dead_Time_from_Onset"] = df[
        "Last_Visit_from_Onset"
    ]
    df.loc[df_event_dead_true.index, "Event_Dead_Time_from_First_Visit"] = df[
        "Last_Visit_from_First_Visit"
    ]
    df_to_update = df.loc[
        (df.Event_Dead == True) & (df.Event_Dead_Delta.notnull())
    ].copy()
    df.loc[df_to_update.index, "Event_Dead_Time_from_Onset_in_days"] = df[
        "Event_Dead_Delta"
    ] + np.abs(df.Symptoms_Onset_Delta)

    in_months = df["Event_Dead_Time_from_Onset_in_days"].apply(
        lambda x: calculate_months_from_days(x)
    )
    df.loc[df_to_update.index, "Event_Dead_Time_from_Onset"] = in_months
    df.loc[df_to_update.index, "Event_Dead_Time_from_First_Visit"] = df[
        "Event_Dead_Delta"
    ].apply(lambda x: calculate_months_from_days(x))
    to_delete = [
        "Event_Dead_Delta",
        "Last_Visit_Delta",
        "Event_Dead_Time_from_Onset_in_days",
    ]
    df.drop(columns=to_delete, inplace=True)
    return df


def preprocess_riluzole(df_to_process, data_dir):
    df = df_to_process.copy()
    data_file = f"{data_dir}/PROACT_RILUZOLE.csv"
    df_raw = pd.read_csv(data_file, delimiter=",")
    df_raw.rename(
        columns={
            "Subject_used_Riluzole": "Riluzole",
            "Riluzole_use_Delta": "Riluzole_Delta",
        },
        inplace=True,
    )
    df_raw = df_raw.groupby("subject_id", as_index=False).agg(
        {
            "Riluzole": "first",
            "Riluzole_Delta": "mean",
        }
    )
    df = join_datasets_by_key(
        df_main=df, df_to_join=df_raw, key_name="subject_id", how="left"
    )

    df.loc[(df.Riluzole.isnull()), "Riluzole"] = "No"
    df.loc[(df.Riluzole_Delta.isnull()), "Riluzole_Delta"] = np.NaN
    df.Riluzole = df.Riluzole.map({"Yes": True, "No": False})
    df["Riluzole_from_Onset_in_days"] = df["Riluzole_Delta"] + np.abs(
        df.Symptoms_Onset_Delta
    )
    in_months = df["Riluzole_from_Onset_in_days"].apply(
        lambda x: calculate_months_from_days(x)
    )
    df["Riluzole_from_Onset"] = in_months
    df["Riluzole_from_First_Visit"] = df["Riluzole_Delta"].apply(
        lambda x: calculate_months_from_days(x, return_abs_value=False)
    )
    df.drop(
        columns=[
            "Riluzole_Delta",
            "Riluzole_from_Onset_in_days",
            "Riluzole_from_Onset",
            "Riluzole_from_First_Visit",
        ],
        inplace=True,
    )
    return df


def preprocess_alsfrs(df_to_process, data_dir):
    df = df_to_process.copy()

    data_file = f"{data_dir}/PROACT_ALSFRS.csv"
    df_raw = pd.read_csv(
        data_file,
        delimiter=",",
        dtype={"Mode_of_Administration": "str", "ALSFRS_Responded_By": "str"},
    )

    to_delete = df_raw.loc[(df_raw.ALSFRS_Delta.isnull())].copy()
    df_raw = remove_rows(df=df_raw, to_delete=to_delete, verbose=True)

    cols_to_remove = ["Mode_of_Administration", "ALSFRS_Responded_By"]
    df_raw.drop(columns=cols_to_remove, inplace=True)

    df_raw["Q5_Cutting"] = np.NaN
    df_raw["Patient_with_Gastrostomy"] = np.NaN
    df_raw["Patient_with_Gastrostomy"] = df_raw["Patient_with_Gastrostomy"].astype(str)

    df_raw.loc[(df_raw["Q5a_Cutting_without_Gastrostomy"].notnull()), "Q5_Cutting"] = (
        df_raw["Q5a_Cutting_without_Gastrostomy"]
    )
    df_raw.loc[
        (df_raw["Q5a_Cutting_without_Gastrostomy"].notnull()),
        "Patient_with_Gastrostomy",
    ] = False

    df_raw.loc[(df_raw["Q5b_Cutting_with_Gastrostomy"].notnull()), "Q5_Cutting"] = (
        df_raw["Q5b_Cutting_with_Gastrostomy"]
    )
    df_raw.loc[
        (df_raw["Q5b_Cutting_with_Gastrostomy"].notnull()),
        "Patient_with_Gastrostomy",
    ] = True

    df_raw.drop(
        columns=["Q5a_Cutting_without_Gastrostomy", "Q5b_Cutting_with_Gastrostomy"],
        axis=1,
        inplace=True,
    )

    to_update = df_raw.loc[
        (df_raw.Q10_Respiratory.isnull()) & (df_raw.R_1_Dyspnea.notnull())
    ].copy()
    df_raw.loc[to_update.index, "Q10_Respiratory"] = df_raw["R_1_Dyspnea"]
    to_delete = df_raw.loc[
        (df_raw.Q1_Speech.isnull())
        | (df_raw.Q2_Salivation.isnull())
        | (df_raw.Q3_Swallowing.isnull())
        | (df_raw.Q4_Handwriting.isnull())
        | (df_raw.Q5_Cutting.isnull())
        | (df_raw.Q6_Dressing_and_Hygiene.isnull())
        | (df_raw.Q7_Turning_in_Bed.isnull())
        | (df_raw.Q8_Walking.isnull())
        | (df_raw.Q9_Climbing_Stairs.isnull())
        | (df_raw.Q10_Respiratory.isnull())
    ].copy()
    df_raw = remove_rows(df=df_raw, to_delete=to_delete, verbose=False)
    df_raw = df_raw[
        [
            "subject_id",
            "Q1_Speech",
            "Q2_Salivation",
            "Q3_Swallowing",
            "Q4_Handwriting",
            "Q5_Cutting",
            "Q6_Dressing_and_Hygiene",
            "Q7_Turning_in_Bed",
            "Q8_Walking",
            "Q9_Climbing_Stairs",
            "Q10_Respiratory",
            "ALSFRS_Delta",
            "Patient_with_Gastrostomy",
        ]
    ].copy()

    df_patients = df[["subject_id", "Symptoms_Onset_Delta"]].copy()

    df_raw = join_datasets_by_key(
        df_main=df_raw, df_to_join=df_patients, key_name="subject_id", raise_error=True
    )

    df_raw.insert(1, "Delta_from_Symptoms_Onset", np.NaN)
    df_raw.insert(2, "Delta_from_First_Visit", np.NaN)

    df_raw["Delta_from_Symptoms_Onset_in_Days"] = df_raw.ALSFRS_Delta + np.abs(
        df_raw.Symptoms_Onset_Delta
    )
    df_raw["Delta_from_Symptoms_Onset"] = np.NaN
    in_months = df_raw["Delta_from_Symptoms_Onset_in_Days"].apply(
        lambda x: calculate_months_from_days(x)
    )
    df_raw.loc[df_raw.index, "Delta_from_Symptoms_Onset"] = in_months
    df_raw["Delta_from_First_Visit_in_Days"] = df_raw.ALSFRS_Delta
    in_months = df_raw["Delta_from_First_Visit_in_Days"].apply(
        lambda x: calculate_months_from_days(x)
    )
    df_raw.loc[df_raw.index, "Delta_from_First_Visit"] = in_months
    df_remove_duplicated = df_raw.sort_values(
        ["subject_id", "Delta_from_First_Visit_in_Days"]
    )
    df_remove_duplicated = (
        df_remove_duplicated.groupby(["subject_id", "Delta_from_First_Visit"])
        .last()
        .reset_index()
    )

    df_raw = df_remove_duplicated.copy()

    to_delete = [
        "Symptoms_Onset_Delta",
        "Delta_from_First_Visit_in_Days",
    ]
    df_raw.drop(columns=to_delete, inplace=True)

    df_raw.sort_values(
        by=["subject_id", "Delta_from_Symptoms_Onset_in_Days"], inplace=True
    )

    cols_to_create = [
        "Q1_Speech",
        "Q2_Salivation",
        "Q3_Swallowing",
        "Q4_Handwriting",
        "Q5_Cutting",
        "Q6_Dressing_and_Hygiene",
        "Q7_Turning_in_Bed",
        "Q8_Walking",
        "Q9_Climbing_Stairs",
        "Q10_Respiratory",
    ]
    max_score = 4

    for col in cols_to_create:
        c = f"Slope_from_Onset_{col}"
        df_raw[c] = np.NaN
        slope = (max_score - df_raw[col]) / df_raw["Delta_from_Symptoms_Onset"]
        df_raw[c] = np.round(slope, 2)

    df_raw["ALSFRS_Total"] = (
        df_raw["Q1_Speech"]
        + df_raw["Q2_Salivation"]
        + df_raw["Q3_Swallowing"]
        + df_raw["Q4_Handwriting"]
        + df_raw["Q5_Cutting"]
        + df_raw["Q6_Dressing_and_Hygiene"]
        + df_raw["Q7_Turning_in_Bed"]
        + df_raw["Q8_Walking"]
        + df_raw["Q9_Climbing_Stairs"]
        + df_raw["Q10_Respiratory"]
    )

    df_raw["Slope_from_Onset_ALSFRS_Total"] = (40 - df_raw["ALSFRS_Total"]) / df_raw[
        "Delta_from_Symptoms_Onset"
    ]

    df_raw["Slope_from_First_Visit_ALSFRS_Total"] = np.NaN
    df_raw["First_Visit_ALSFRS_Total"] = np.NaN

    subject_ids = df_raw.subject_id.unique()
    for subject_id in subject_ids:
        current_patient = df_raw.loc[(df_raw.subject_id == subject_id)]
        first_visit = current_patient.loc[
            (current_patient.Delta_from_First_Visit == 0.0)
        ]
        if first_visit.shape[0] > 0:
            df_raw.loc[first_visit.index, "Slope_from_First_Visit_ALSFRS_Total"] = 0.0
            first_visit_score = first_visit["ALSFRS_Total"].values[0]
            to_calculate = current_patient.loc[
                (current_patient.Delta_from_First_Visit > 0.0)
            ].copy()
            df_raw.loc[to_calculate.index, "Slope_from_First_Visit_ALSFRS_Total"] = (
                df_raw["ALSFRS_Total"] - first_visit_score
            ) / df_raw["Delta_from_First_Visit"]
            df_raw.loc[current_patient.index, "First_Visit_ALSFRS_Total"] = (
                first_visit_score
            )

            for col in cols_to_create:
                c = f"Slope_from_First_Visit_{col}"
                df_raw.loc[first_visit.index, c] = 0.0
                first_visit_score = first_visit[col].values[0]
                df_raw.loc[to_calculate.index, c] = (
                    df_raw[col] - first_visit_score
                ) / df_raw["Delta_from_First_Visit"]

    df_raw["Region_Involved_Bulbar"] = (
        (df_raw["Q1_Speech"] + df_raw["Q2_Salivation"] + df_raw["Q3_Swallowing"]) < 12
    ) * 1.0

    df_raw["Region_Involved_Upper_Limb"] = (
        (df_raw["Q4_Handwriting"] + df_raw["Q5_Cutting"]) < 8
    ) * 1.0

    df_raw["Region_Involved_Lower_Limb"] = (
        (df_raw["Q8_Walking"] + df_raw["Q9_Climbing_Stairs"]) < 8
    ) * 1.0

    df_raw["Region_Involved_Respiratory"] = ((df_raw["Q10_Respiratory"]) < 4) * 1.0

    df_raw["Qty_Regions_Involved"] = np.NaN

    df_raw["Qty_Regions_Involved"] = (
        df_raw["Region_Involved_Bulbar"]
        + df_raw["Region_Involved_Upper_Limb"]
        + df_raw["Region_Involved_Lower_Limb"]
        + df_raw["Region_Involved_Respiratory"]
    )

    return df_raw


def generate_time_series_disease_duration_based_on_alsfrs(df_temporal):
    n_years = 10
    threshold = 12 * n_years
    months = np.linspace(0, threshold, threshold + 1, dtype=float)
    col_baseline = "Delta_from_First_Visit"
    column = "Delta_from_Symptoms_Onset"
    df_copy = df_temporal.loc[(df_temporal[col_baseline] == 0.0)].copy()
    df_copy.dropna(
        subset=[
            col_baseline,
            column,
        ],
        inplace=True,
    )
    df_pivot = df_copy.copy()
    df_aux = df_pivot.pivot_table(
        index="subject_id", columns=col_baseline, values=column, aggfunc="max"
    )
    df_aux.reset_index(inplace=True)
    df_data = {"subject_id": df_aux["subject_id"], 0.0: df_aux[0.0].values}
    for month in months[1:]:
        df_data[month] = df_data[month - 1] + 1
    df_aux = pd.DataFrame(df_data)
    df_to_save = df_aux.dropna(axis=0, how="all")
    df_aux.loc[df_to_save.index].to_csv(
        "Data/preprocessed/Disease_Duration.csv", index=False
    )
    return sorted(df_aux.subject_id.unique())


def generate_time_series_alsfrs(df_temporal):
    n_years = 10
    threshold = 12 * n_years
    months = np.linspace(0, threshold, threshold + 1, dtype=float)

    columns_questions_ALSFRS = [
        "Q1_Speech",
        "Q2_Salivation",
        "Q3_Swallowing",
        "Q4_Handwriting",
        "Q5_Cutting",
        "Q6_Dressing_and_Hygiene",
        "Q7_Turning_in_Bed",
        "Q8_Walking",
        "Q9_Climbing_Stairs",
        "Q10_Respiratory",
    ]

    columns = [
        "Q1_Speech",
        "Q2_Salivation",
        "Q3_Swallowing",
        "Q4_Handwriting",
        "Q5_Cutting",
        "Q6_Dressing_and_Hygiene",
        "Q7_Turning_in_Bed",
        "Q8_Walking",
        "Q9_Climbing_Stairs",
        "Q10_Respiratory",
        "Region_Involved_Bulbar",
        "Region_Involved_Upper_Limb",
        "Region_Involved_Lower_Limb",
        "Region_Involved_Respiratory",
        "Qty_Regions_Involved",
        "Patient_with_Gastrostomy",
        "ALSFRS_Total",
    ]

    for column in columns:
        col_baseline = "Delta_from_First_Visit"
        df_copy = df_temporal.sort_values(by=["subject_id", col_baseline]).copy()
        if (column == "Patient_with_Gastrostomy") | (
            column.startswith("Region_Involved_")
        ):
            df_copy[column].replace({True: 1, False: 0}, inplace=True)
        df_copy.dropna(
            subset=[
                col_baseline,
                column,
            ],
            inplace=True,
        )
        df_pivot = df_copy.copy()
        df_aux = df_pivot.pivot_table(
            index="subject_id", columns=col_baseline, values=column, aggfunc="max"
        )
        df_aux.reset_index(inplace=True)
        cols_months = df_aux.columns[1:]
        for month in months:
            if month not in cols_months:
                df_aux.insert(int(month), month, np.NaN)
        cols_months_ordered = list(sorted(months))
        cols_months_ordered.insert(0, "subject_id")
        df_aux = df_aux[cols_months_ordered]
        round_decimal_places = 2
        if (column == "Slope_from_Onset_Total_Score") | (
            "Slope_from_Onset_Q" in column
        ):
            pass
        elif column in columns_questions_ALSFRS:
            round_decimal_places = 0
        elif column == "ALSFRS_Total":
            round_decimal_places = 0
        elif column in columns:
            round_decimal_places = 0
        col_name = column.replace("_from_Onset", "")
        df_fill_nan_using_interpolation = df_aux
        cols_months = df_fill_nan_using_interpolation.columns[1:]
        df_to_save = df_aux[df_aux.columns[1:]].dropna(axis=0, how="all")
        if col_name == "Q1_Speech":
            df_aux.loc[df_to_save.index].to_csv(
                f"Data/preprocessed/{col_name}_RAW.csv", index=False
            )
        df_aux = (
            df_fill_nan_using_interpolation[cols_months]
            .interpolate(
                method="linear",
                limit_direction="both",
                limit=1000,
                axis=1,
                inplace=False,
            )
            .copy()
        )
        df_aux[cols_months] = np.round(df_aux[cols_months], round_decimal_places)
        df_fill_nan_using_interpolation[cols_months] = df_aux[cols_months]
        df_fill_nan_using_interpolation.dropna(inplace=True)
        csv_file = f"Data/preprocessed/{col_name}.csv"
        df_fill_nan_using_interpolation.to_csv(csv_file, index=False)
        df_aux = df_fill_nan_using_interpolation.copy()
        print(f"{col_name} :: {df_aux.shape}")


def get_first_last_visits(df_patients, df_alsfrs, max_months_to_analyze=12):
    patients_with_visits_enough = (
        df_patients.loc[(df_patients.Qty_Measurements_ALSFRS >= 3)]
        .copy()
        .subject_id.unique()
    )
    df_alsfrs = df_alsfrs.loc[
        (df_alsfrs.subject_id.isin(patients_with_visits_enough))
    ].copy()
    selected_patients_ids = []

    for subject_id in patients_with_visits_enough:
        df_current_patient = df_alsfrs.loc[(df_alsfrs.subject_id == subject_id)].copy()
        df_current_patient.dropna(axis=1, inplace=True)
        try:
            month_of_last_visit = float(df_current_patient.columns[-1])
        except Exception:
            continue
        if month_of_last_visit >= max_months_to_analyze:
            id = df_current_patient.subject_id.values[0]
            selected_patients_ids.append(id)
    df_return = df_patients.loc[(df_patients.subject_id.isin(selected_patients_ids))]
    return df_return


def perform_data_codification(df_patients):
    df_coded = df_patients.copy()
    df_coded.Sex = df_coded.Sex.map({"Male": 1, "Female": 0})
    df_coded.rename(columns={"Sex": "Sex_Male"}, inplace=True)
    df_coded.Site_Onset = df_coded.Site_Onset.map(
        {"Bulbar": 0, "Limb/Spinal": 1, "Other": 2}
    )
    df_coded.Riluzole = df_coded.Riluzole.map({True: 1, False: 0})
    df_coded.Event_Dead = df_coded.Event_Dead.map({True: 1, False: 0})
    return df_coded


def generate_time_series_static_features_based_on_alsfrs(df_temporal, df_patients):
    n_years = 10
    threshold = 12 * n_years
    months = np.linspace(0, threshold, threshold + 1, dtype=float)
    df_patients_aux = df_patients.loc[
        (df_patients.subject_id.isin(df_temporal.subject_id.unique()))
    ].copy()
    df_static_as_ts = None
    for column in df_patients_aux.columns:
        if column != "subject_id":
            df_aux = df_patients_aux[["subject_id", column]].copy()
            for month in months:
                df_aux[str(month)] = df_aux[column].copy()
            df_aux.insert(1, "feature", column)
            df_aux.drop([column], axis="columns", inplace=True)
            df_static_as_ts = (
                df_aux
                if df_static_as_ts is None
                else pd.concat([df_static_as_ts, df_aux], ignore_index=True)
            )
    return df_static_as_ts
