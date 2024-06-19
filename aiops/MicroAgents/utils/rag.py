import polars as pl



class RAG:
    def __init__(self):
        self.df = pl.DataFrame({
            'KEY SYMPTOM':[],
            'SECONDARY SYMPTOM':[],
            'ROOT CAUSE':[],
            'CAUSE TYPE':[],
            'PODNAME':[],
            'DESCRIPTION':[]
        }, {
            'KEY SYMPTOM':pl.Utf8,
                    'SECONDARY SYMPTOM':pl.Utf8,
                    'ROOT CAUSE':pl.Utf8,
                    'CAUSE TYPE':pl.Utf8,
                    'PODNAME':pl.Utf8,
                    'DESCRIPTION':pl.Utf8
        })
    def save(self, incident):
        new_row = pl.DataFrame(incident)
        self.df = self.df.vstack(new_row)
        return self.df
    
    def cal_jaccard(self, symptom1, symptom2):
        return len(set(symptom1).intersection(set(symptom2))) / len(set(symptom1).union(set(symptom2)))

    def retrieve(self, symptoms):
        res = []
        # print(symptoms, self.df)
        for row in self.df.iter_rows(named=True):
            if row['KEY SYMPTOM'] in symptoms:
                secondary_row = row['SECONDARY SYMPTOM'].split('<STOP>')
                row['score'] = self.cal_jaccard(symptoms, secondary_row)
                res.append(row)
        res.sort(key=lambda x: x['score'] , reverse=True)
        return res
    
    def retrieve_to_text(self, res, k=3):
        
        res.sort(key=lambda x: x['score'], reverse=True)        
        res = res[:k]

        res_text = ''
        for i, temp_res in enumerate(res):
            
            res_text += f"""INCIDENT {i+1}: \nSYMPTOM: {temp_res['DESCRIPTION']}, \nCAUSE: [\n{{"MODULE":{temp_res['ROOT CAUSE']},\n"TYPE":{temp_res['CAUSE TYPE']},\n"Possibility": High}}\n]\n"""
        return res_text

    