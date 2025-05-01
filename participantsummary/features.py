import pandas as pd
from radarpipeline.features import Feature, FeatureGroup


class ParticipantSummaryActiveData(FeatureGroup):
    def __init__(self):
        name = "ParticipantSummaryActiveData"
        description = "Participant Summary Feature Group for Active Data"
        features = [NumberOfQuestionnaireComplete,
                    NumOfDifferentTypesOfMedication,
                    BAARSSymptomsSummary, PHQ8SymptomsSummary,
                    BloodPressureSummary, MedicationTimeline]
        super().__init__(name, description, features)

    def preprocess(self, data):
        """
        Preprocess the data for each feature in the group.
        """
        return data


class MedicationTimeline(Feature):
    def __init__(self):
        self.name = "MedicationTimeline"
        self.description = "Medication Timeline"
        self.required_input_data = ["questionnaire_adhd_medication_use_daily", "questionnaire_response/ADHDDailyMedicationUse"]

    def preprocess(self, data):
        return data

    def flatten_dict(self, row):
        key_value_dicts = {}
        num_indx = 0
        while f'value.answers.{num_indx}.questionId' in row and row[f'value.answers.{num_indx}.questionId'] is not None:
            key_value_dicts.update(dict(zip([row[f'value.answers.{num_indx}.questionId']], [row[f'value.answers.{num_indx}.value']])))
            num_indx += 1
        return key_value_dicts

    def get_first_medication_date(self, row):
        med_rows = row[row['meduse_1'] == 1].reset_index(drop=True)
        return med_rows[['value.time','meduse_1', 'meduse_2a_1',  'meduse_2a_1o', 'meduse_2b_1', 'meduse_2d_1']]

    def calculate(self, data) -> pd.DataFrame:
        medication_dfs = data.get_variable_data(self.required_input_data)
        medication_df = pd.concat(medication_dfs, axis=0).reset_index(drop=True)
        medication_df = medication_df.sort_values(by=['key.userId', 'value.time']).reset_index(drop=True)
        medication_df['json_value_pair'] = medication_df.apply(self.flatten_dict, axis=1)
        medication_df = medication_df[['key.projectId', 'key.userId', 'value.time', 'value.timeCompleted', 'json_value_pair']]
        medication_df_final = pd.concat([medication_df, pd.json_normalize(medication_df['json_value_pair'])], axis=1)
        df_medication_summary = medication_df_final.groupby('key.userId').apply(self.get_first_medication_date)
        df_medication_summary.reset_index(inplace=True)
        return df_medication_summary

class NumberOfQuestionnaireComplete(Feature):
    def __init__(self):
        self.name = "NumberOfQuestionnaireComplete"
        self.description = "Number of Questionnaires Completed"

        # List of questionnaire variables in questionnaire response
        self.questionnaire_response_variables = [
            "questionnaire_response/ADHDMedicationUseSideEffects",
            "questionnaire_response/WeightAndWaistCircumference",
            "questionnaire_response/BloodPressureMeasurement",
            "questionnaire_response/FND",
            "questionnaire_response/Audit",
            "questionnaire_response/MediterraneanDietAdherence",
            "questionnaire_response/RPQ-A",
            "questionnaire_response/ADHDDailyMedicationUse",
            "questionnaire_response/ARI_SELF",
            "questionnaire_response/PHQ8",
            "questionnaire_response/GAD7",
            "questionnaire_response/BAARS-IV",
        ]
        self.variable_mapping = {
            "questionnaire_response/ADHDMedicationUseSideEffects": "questionnaire_adhd_medication_side_effects",
            "questionnaire_response/WeightAndWaistCircumference": "questionnaire_weight_and_waist_circumference",
            "questionnaire_response/BloodPressureMeasurement": "questionnaire_blood_pressure_measurement",
            "questionnaire_response/FND": "questionnaire_fnd",
            "questionnaire_response/Audit": "questionnaire_audit",
            "questionnaire_response/MediterraneanDietAdherence": "questionnaire_mediterranean_diet_adherence",
            "questionnaire_response/RPQ-A": "questionnaire_rpq_a",
            "questionnaire_response/ADHDDailyMedicationUse": "questionnaire_adhd_medication_use_daily",
            "questionnaire_response/ARI_SELF": "questionnaire_ari_self",
            "questionnaire_response/PHQ8": "questionnaire_adhd_phq8",
            "questionnaire_response/GAD7": "questionnaire_gad7",
            "questionnaire_response/BAARS-IV": "questionnaire_baars_iv"
        }

        self.required_input_data = [
            "questionnaire_adhd_medication_use_daily",
            "questionnaire_baars_iv",
            "questionnaire_adhd_medication_side_effects",
            "questionnaire_blood_pressure_measurement",
            "questionnaire_weight_and_waist_circumference",
            "questionnaire_fnd",
            "questionnaire_audit",
            "questionnaire_mediterranean_diet_adherence",
            "questionnaire_adhd_phq8",
            "questionnaire_gad7",
            "questionnaire_rpq_a"] + self.questionnaire_response_variables

    def preprocess(self, data):
        questionnaire_dict = {}
        questionnaire_data = data.get_combined_data_by_variable(
            self.required_input_data,
            return_dict=True
        )
        questionnaire_dict = {}
        for variable in self.variable_mapping:
            if self.variable_mapping[variable] in questionnaire_data and variable in questionnaire_data:
                questionnaire_dict[self.variable_mapping[variable]] = pd.concat(
                    [questionnaire_data[self.variable_mapping[variable]],
                     questionnaire_data[variable]], axis=0)
            elif variable in questionnaire_data:
                questionnaire_dict[self.variable_mapping[variable] ] = questionnaire_data[variable]
            elif self.variable_mapping[variable] in questionnaire_data:
                questionnaire_dict[self.variable_mapping[variable]] = questionnaire_data[
                    self.variable_mapping[variable]]
            else:
                questionnaire_dict[variable] = pd.DataFrame()
        return questionnaire_dict

    def calculate(self, questionnaire_dict) -> pd.DataFrame:
        """
        Calculate the number of questionnaires completed.
        """
        questionnaire_count_dfs = []
        for var in self.variable_mapping:
            variable = self.variable_mapping[var]
            if variable not in questionnaire_dict:
                continue
            p = questionnaire_dict[variable][['key.projectId',
                                              'key.userId']].groupby(
                                                  'key.userId'
                                                  ).count().reset_index()
            p.rename({'key.projectId': f"{variable}_count"}, axis=1,
                     inplace=True)
            questionnaire_count_dfs.append(p)
        dfs = [df.set_index('key.userId') for df in questionnaire_count_dfs]
        return pd.concat(dfs, axis=1).reset_index().reset_index()


class NumOfDifferentTypesOfMedication(Feature):
    def __init__(self):
        self.name = "NumOfDifferentTypesOfMedication"
        self.description = "Number of Different Types of Medication"
        self.required_input_data = ["questionnaire_adhd_medication_use_daily", "questionnaire_response/ADHDDailyMedicationUse"]

    def flatten_dict(self, row):
        key_value_dicts = {}
        num_indx = 0
        while f'value.answers.{num_indx}.questionId' in row and row[f'value.answers.{num_indx}.questionId'] is not None:
            key_value_dicts.update(dict(zip([row[f'value.answers.{num_indx}.questionId']], [row[f'value.answers.{num_indx}.value']])))
            num_indx += 1
        return key_value_dicts

    def flatten_df(self, df):
        # Flatten the DataFrame
        df['json_value_pair'] = df.apply(self.flatten_dict, axis=1)
        df = df[['key.projectId', 'key.userId', 'value.time',
                 'value.timeCompleted', 'json_value_pair']]
        df = pd.concat([df, pd.json_normalize(df['json_value_pair'])], axis=1)
        df.drop(columns=['json_value_pair'], inplace=True)
        return df

    def preprocess(self, data):
        return data

    def calculate(self, data) -> pd.DataFrame:
        ques_dfs = data.get_variable_data(self.required_input_data)
        # concat
        for i, ques_df in enumerate(ques_dfs):
            # flatten the DataFrame
            ques_dfs[i] = self.flatten_df(ques_df)
        ques_df = pd.concat(ques_dfs, axis=0)
        ques_df.reset_index(drop=True, inplace=True)
        # filter out rows where 'value.answers.5.value' is NaN
        ques_df = ques_df.dropna(subset=['meduse_2a_1'])
        ques_df_unq = ques_df.groupby('key.userId')[
            'meduse_2a_1'].unique().reset_index()
        ques_df_unq['unique_medications'] = ques_df_unq[
            'meduse_2a_1'].apply(lambda x: len(x))
        return ques_df_unq


class BAARSSymptomsSummary(Feature):
    def __init__(self):
        self.name = "BAARSSymptomsSummary"
        self.description = "Summary of BAARS Symptoms"
        self.required_input_data = ["questionnaire_baars_iv", "questionnaire_response/BAARS-IV"]

    def flatten_dict(self, row):
        key_value_dicts = {}
        num_indx = 0
        while f'value.answers.{num_indx}.questionId' in row and row[f'value.answers.{num_indx}.questionId'] is not None:
            key_value_dicts.update(dict(zip([row[f'value.answers.{num_indx}.questionId']], [row[f'value.answers.{num_indx}.value']])))
            num_indx += 1
        return key_value_dicts

    def flatten_df(self, df):
        # Flatten the DataFrame
        df['json_value_pair'] = df.apply(self.flatten_dict, axis=1)
        df = df[['key.projectId', 'key.userId', 'value.time',
                 'value.timeCompleted', 'json_value_pair']]
        df = pd.concat([df, pd.json_normalize(df['json_value_pair'])], axis=1)
        df.drop(columns=['json_value_pair'], inplace=True)
        return df

    def preprocess(self, data):
        return data

    def calculate(self, data) -> pd.DataFrame:
        baars_dfs = data.get_variable_data(self.required_input_data)
        # flatten the DataFrame
        for i, baars_df in enumerate(baars_dfs):
            baars_dfs[i] = self.flatten_df(baars_df)
        baars_df = pd.concat(baars_dfs, axis=0)
        return baars_df


class PHQ8SymptomsSummary(Feature):
    def __init__(self):
        self.name = "PHQ8SymptomsSummary"
        self.description = "Summary of PHQ-8 Symptoms"
        self.required_input_data = ["questionnaire_adhd_phq8", "questionnaire_response/PHQ8"]

    def flatten_dict(self, row):
        key_value_dicts = {}
        num_indx = 0
        while f'value.answers.{num_indx}.questionId' in row and row[f'value.answers.{num_indx}.questionId'] is not None:
            key_value_dicts.update(dict(zip([row[f'value.answers.{num_indx}.questionId']], [row[f'value.answers.{num_indx}.value']])))
            num_indx += 1
        return key_value_dicts

    def flatten_df(self, df):
        # Flatten the DataFrame
        df['json_value_pair'] = df.apply(self.flatten_dict, axis=1)
        df = df[['key.projectId', 'key.userId', 'value.time',
                 'value.timeCompleted', 'json_value_pair']]
        df = pd.concat([df, pd.json_normalize(df['json_value_pair'])], axis=1)
        df.drop(columns=['json_value_pair'], inplace=True)
        return df

    def preprocess(self, data):
        return data

    def calculate(self, data) -> pd.DataFrame:
        phq8_dfs = data.get_variable_data(self.required_input_data)
        # flatten the DataFrame
        for i, phq8_df in enumerate(phq8_dfs):
            phq8_dfs[i] = self.flatten_df(phq8_df)
        phq8_df = pd.concat(phq8_dfs, axis=0)
        return phq8_df


class BloodPressureSummary(Feature):
    def __init__(self):
        self.name = "BloodPressureSummary"
        self.description = "Summary of Blood Pressure"
        self.required_input_data = ["questionnaire_blood_pressure_measurement", "questionnaire_response/BloodPressureMeasurement"]

    def flatten_dict(self, row):
        key_value_dicts = {}
        num_indx = 0
        while f'value.answers.{num_indx}.questionId' in row and row[f'value.answers.{num_indx}.questionId'] is not None:
            key_value_dicts.update(dict(zip([row[f'value.answers.{num_indx}.questionId']], [row[f'value.answers.{num_indx}.value']])))
            num_indx += 1
        return key_value_dicts

    def flatten_df(self, df):
        # Flatten the DataFrame
        df['json_value_pair'] = df.apply(self.flatten_dict, axis=1)
        df = df[['key.projectId', 'key.userId', 'value.time',
                 'value.timeCompleted', 'json_value_pair']]
        df = pd.concat([df, pd.json_normalize(df['json_value_pair'])], axis=1)
        df.drop(columns=['json_value_pair'], inplace=True)
        return df

    def preprocess(self, data):
        return data

    def calculate(self, data) -> pd.DataFrame:
        bp_dfs = data.get_variable_data(self.required_input_data)
        # flatten the DataFrame
        for i, bp_df in enumerate(bp_dfs):
            bp_dfs[i] = self.flatten_df(bp_df)
        bp_df = pd.concat(bp_dfs, axis=0)
        return bp_df


class ParticipantSummaryPassiveData(FeatureGroup):
    def __init__(self):
        name = "ParticipantSummaryPassiveData"
        description = "Participant Summary Feature Group for Passive Data"
        features = [AverageTimeSpentonPhone,
                    MostOpenedApps,
                    NumberOfDifferentAppsUsed]
        super().__init__(name, description, features)

    def preprocess(self, data):
        """
        Preprocess the data for each feature in the group.
        """
        return data


class AverageTimeSpentonPhone(Feature):
    def __init__(self):
        self.name = "AverageTimeSpentonPhone"
        self.description = "Average Time Spent on Phone"
        self.required_input_data = ["android_phone_user_interaction"]

    def preprocess(self, data):
        df_user_interaction = data.get_combined_data_by_variable(
            "android_phone_user_interaction"
        )
        df_user_interaction['value.time'] = pd.to_datetime(df_user_interaction['value.time'], unit="s")
        df_user_interaction['value.timeReceived'] = pd.to_datetime(df_user_interaction['value.timeReceived'], unit="s")
        df_user_interaction['key.userId'] = df_user_interaction['key.userId'].str.strip()
        # Removing duplicates
        df_user_interaction = df_user_interaction[~df_user_interaction[["key.userId", "value.time", "value.interactionState"]].duplicated()]
        df_user_interaction.reset_index(drop=True, inplace=True)
        df_user_interaction["previous_interactionState"] = df_user_interaction.groupby("key.userId")['value.interactionState'].shift()
        df_user_interaction["previous_interactionState_time"] = df_user_interaction.groupby("key.userId")['value.time'].shift()
        df_unlocked_duration = df_user_interaction[df_user_interaction["previous_interactionState"]=="UNLOCKED"].reset_index(drop=True)
        df_unlocked_duration = df_unlocked_duration.rename({"key.userId":"uid",
                                                            "value.time":"time",
                                                            "value.interactionState":"interactionState"},
                                                           axis=1)
        df_unlocked_duration["begin_time"] = df_unlocked_duration["previous_interactionState_time"].astype('datetime64[s]')
        df_unlocked_duration["end_time"] = df_unlocked_duration["time"].astype('datetime64[s]')
        df_active_session_details = df_unlocked_duration[['key.projectId', 'uid', 'key.sourceId',
       'begin_time', 'end_time']]
        return df_active_session_details

    def calculate(self, df_active_session_details) -> pd.DataFrame:
        df_active_session_details['duration'] = df_active_session_details['end_time'] - df_active_session_details['begin_time']
        df_active_session_details['duration_seconds'] = df_active_session_details['duration'].dt.total_seconds()
        df_active_session_details = df_active_session_details[df_active_session_details['duration_seconds'] > 0].reset_index(drop=True)
        df_active_session_details = df_active_session_details.groupby("uid")["duration_seconds"].mean().reset_index()
        return df_active_session_details


class MostOpenedApps(Feature):
    def __init__(self):
        self.name = "MostOpenedApps"
        self.description = "Most Opened Apps"
        self.required_input_data = ["android_phone_usage_event"]

    def preprocess(self, data):
        return data

    def calculate(self, data) -> pd.DataFrame:
        df = data.get_variable_data("android_phone_usage_event")
        df_most_freq_app = df.groupby('key.userId')['value.packageName'].value_counts().reset_index()
        return df_most_freq_app


class NumberOfDifferentAppsUsed(Feature):
    def __init__(self):
        self.name = "NumberOfDifferentAppsUsed"
        self.description = "Number of Different Apps Used"
        self.required_input_data = ["android_phone_usage_event"]

    def preprocess(self, data):
        return data

    def calculate(self, data) -> pd.DataFrame:
        df = data.get_variable_data("android_phone_usage_event")
        df_unique_apps = df.groupby('key.userId')[
            'value.packageName'].nunique().reset_index()
        df_unique_apps = df_unique_apps.rename(
            columns={"value.packageName": "num_unique_apps"})
        return df_unique_apps
