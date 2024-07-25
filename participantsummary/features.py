import pandas as pd
from radarpipeline.features import Feature, FeatureGroup


class ParticipantSummaryActiveData(FeatureGroup):
    def __init__(self):
        name = "ParticipantSummaryActiveData"
        description = "Participant Summary Feature Group for Active Data"
        features = [NumberOfQuestionnaireComplete,
                    NumOfDifferentTypesOfMedication,
                    BAARSSymptomsSummary, PHQ8SymptomsSummary,
                    BloodPressureSummary]
        super().__init__(name, description, features)

    def preprocess(self, data):
        """
        Preprocess the data for each feature in the group.
        """
        return data


class NumberOfQuestionnaireComplete(Feature):
    def __init__(self):
        self.name = "NumberOfQuestionnaireComplete"
        self.description = "Number of Questionnaires Completed"
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
            "questionnaire_gad7"]

    def preprocess(self, data):
        questionnaire_dict = {}
        for variable in self.required_input_data:
            questionnaire_dict[variable] = data.get_variable_data(variable)
        return questionnaire_dict


    def calculate(self, questionnaire_dict) -> pd.DataFrame:
        """
        Calculate the number of questionnaires completed.
        """
        questionnaire_count_dfs = []
        for variable in self.required_input_data:
            p = questionnaire_dict[variable][['key.projectId',
                                              'key.userId']].groupby('key.userId').count().reset_index()
            p.rename({'key.projectId':f"{variable}_count"}, axis=1, inplace=True)
            questionnaire_count_dfs.append(p)
        dfs = [df.set_index('key.userId') for df in questionnaire_count_dfs]
        return pd.concat(dfs, axis=1)


class NumOfDifferentTypesOfMedication(Feature):
    def __init__(self):
        self.name = "NumOfDifferentTypesOfMedication"
        self.description = "Number of Different Types of Medication"
        self.required_input_data = ["questionnaire_adhd_medication_use_daily"]


    def preprocess(self, data):
        return data

    def calculate(self, data) -> pd.DataFrame:
        ques_df = data.get_variable_data("questionnaire_adhd_medication_use_daily")
        ques_df = ques_df.dropna(subset=['value.answers.5.value'])
        ques_df_unq = ques_df.groupby('key.userId')['value.answers.5.value'].unique().reset_index()
        ques_df_unq['unique_medications'] = ques_df_unq['value.answers.5.value'].apply(lambda x: len(x))
        return ques_df_unq


class BAARSSymptomsSummary(Feature):
    def __init__(self):
        self.name = "BAARSSymptomsSummary"
        self.description = "Summary of BAARS Symptoms"
        self.required_input_data = ["questionnaire_baars_iv"]


    def preprocess(self, data):
        return data

    def calculate(self, data) -> pd.DataFrame:
        baars_df = data.get_variable_data("questionnaire_baars_iv")
        return baars_df


class PHQ8SymptomsSummary(Feature):
    def __init__(self):
        self.name = "PHQ8SymptomsSummary"
        self.description = "Summary of PHQ-8 Symptoms"
        self.required_input_data = ["questionnaire_adhd_phq8"]

    def preprocess(self, data):
        return data

    def calculate(self, data) -> pd.DataFrame:
        phq8_df = data.get_variable_data("questionnaire_adhd_phq8")
        return phq8_df


class BloodPressureSummary(Feature):
    def __init__(self):
        self.name = "BloodPressureSummary"
        self.description = "Summary of Blood Pressure"
        self.required_input_data = ["questionnaire_blood_pressure_measurement"]

    def preprocess(self, data):
        return data

    def calculate(self, data) -> pd.DataFrame:
        bp_df = data.get_variable_data("questionnaire_blood_pressure_measurement")
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
            columns={"value.packageName": "num_unique_apps"}, axis=1)
        return df_unique_apps
