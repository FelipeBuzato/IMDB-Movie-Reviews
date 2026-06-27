class PreProcessor:
    def __init__(self, data):
        self.data = data
        column_groups = self.split_columns()
        self.nlp_columns = column_groups[0]


    def split_columns(self):
        all_columns = self.data.columns.tolist()
        all_columns.remove('id')
        if('sentiment' in all_columns):
            all_columns.remove('sentiment')

        nlp_columns = all_columns
        return [nlp_columns]