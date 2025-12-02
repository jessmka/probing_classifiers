import pandas as pd
import numpy as np
import re
import json
import torch
import random
from probes import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
# torch.set_printoptions(profile="full")
import re
import pandas as pd
# pd.set_option('display.max_columns', None)


class MappingDicts:
    def __init__(self, folder):
        self.user_id_map = pd.read_csv(os.path.join(folder,'user_id_map.csv'))
        self.book_id_map = pd.read_csv(os.path.join(folder,'book_id_map.csv'))
        self.author_gender = pd.read_csv('final_dataset.csv')
        with open(os.path.join(folder, 'gender_dict_500.json'), 'r') as f:
            self.gender_dict = json.load(f)
        with open(os.path.join(folder,'author_data.json'), 'r') as f:
            self.author_dict = json.load(f)
        with open(os.path.join(folder,'book_data.json'), 'r') as f:
            self.book_dict = json.load(f)


class RecData:
    def __init__(self, foldername, filename='goodreads_samples2.csv'):
        self.map_dicts = MappingDicts(foldername)
        self.df = pd.read_csv(os.path.join(foldername,filename))

    def merged(self, lo_rating=4, num_ratings=3):
        merged_df = pd.merge(
            self.df, 
            self.map_dicts.book_id_map, 
            left_on ='book_id',
            right_on='book_id_csv', 
            how='left'
            )
        totals = merged_df.groupby('user_id')['rating'].agg(lambda x: (x >= lo_rating).sum()).reset_index()
        totals.columns = ['user_id','rating_count']
        merged_df2 = pd.merge(
            merged_df, 
            totals[totals.rating_count >= num_ratings], 
            on ='user_id', 
            how='inner'
            )
        return merged_df2.groupby('user_id')['book_id_y'].agg(list).reset_index()
    
    def user_title_dict(self, result_df):
        # Returns a mapping dict with the users total historical titles read and liked 
        user_title_dict = {}
        for i, row in result_df.iterrows():
            user_title_dict[row['user_id']] = [self.map_dicts.book_dict[str(i)]['title_without_series'] for i in row['book_id_y']]
        return user_title_dict
    
class LLMRecs:
    def __init__(self, foldername, merged, user_title_dict, filename='output_dict_500.pt'):
        self.outputs = torch.load(os.path.join(foldername, filename), map_location=torch.device('cpu'))
        self.map_dicts = MappingDicts(foldername)
        self.book_dict_reverse = {v['title']: k for k, v in self.map_dicts.book_dict.items()}
        self.merged = merged
        self.user_title_dict = user_title_dict
        self.str1 = "system\n\nCutting Knowledge Date: December 2023\nToday Date: 08 Nov 2025\n\nuser\n\nHi, I\'ve read and enjoyed the following books:"
        self.str1b = "Hi, I\'ve read and enjoyed the following books:"
        self.str2 = """  Only return the 5 books you recommend in JSON format like {"Books": {\'title\':..., \'author\':...}}, and nothing else.assistant\n\n"""
        self.str3 = """Please recommend new books based on the user\'s reading preferences and only return the 5 books you recommend in JSON format like {"Books": {\'title\':..., \'author\':...}}, and nothing else.assistant"""
        

    def extract_all_titles(self, text):
        raw_titles = re.findall(r'"title"\s*:\s*"([^"]+)"', text)
        cleaned = [t.split(" by ")[0].strip() for t in raw_titles]
        return cleaned
    
    def extract_books(self, entry):
        # Case 1: structured JSON-like string
        if isinstance(entry, dict) and "Books" in entry:
            return entry["Books"]
        
        if isinstance(entry, str) and entry.strip().startswith("{"):
            try:
                data = json.loads(entry)
                if "Books" in data:
                    return data["Books"]
            except json.JSONDecodeError:
                pass  # fall through to regex

        # Case 2: free-form text like "Title (Series, #N) by Author"
        pattern = r"([A-Z][^()]*?(?:\([^)]*\))?\s+by\s+[A-Z][^,\.]+)"
        matches = re.findall(pattern, entry)

        books = []
        for m in matches:
            # Split into title and author
            title_part, author = m.split(" by ", 1)
            # try to merge book id in
            if title_part.strip() in self.book_dict_reverse:
                book_id = self.book_dict_reverse[title_part.strip()]
            else:
                book_id = None
            books.append({"title": title_part.strip(), "author": author.strip(), "book_id": book_id})
        
        return books
    
    def small_dict(self, df=True):
        """ Returns a version of the outputs without the hidden state repr"""
        small_dict = {}
        for k, v in self.outputs.items():
            inner_dict = {prop:value for prop, value in v.items() if ((prop != 'pre_hidden') and (prop != 'post_hidden'))}
            small_dict[k] = inner_dict
            

        if df:
            small_df = pd.DataFrame.from_dict(small_dict, orient='index').reset_index().rename(columns={'index':'user_id'})
            return small_df
        else: 
            return small_dict    
    



    def map_titles(self, df):
        """ Tries to parse LLM output text  and map to titles in existing book dict"""
        # for k, values in self.outputs.items():
        for typ in ('baseline', 'steered'):
            df[f'{typ}_recs'] = df[f'{typ}_text'].str.split(self.str3).str[1].str.replace("\n","")
            df[f'{type}_titles'] = df[f'{typ}_recs'].apply(self.extract_all_titles)
            df[f'{typ}_ids'] = df[f'{typ}_titles'].apply(
                lambda x: [
                    self.book_dict_reverse[item] for item in x if item in self.book_dict_reverse
                    ]
                )
        # And for the original sampled books
        df['hist'] = df['baseline_text'].str.split(self.str3).str[0].str.split(self.str1b).str[1].str.replace(self.str1,"").apply(extract_books)
        df['hist_titles'] = df['hist'].apply(lambda x: [i['title'] for i in x])
        
    def count_from_hist(self, df):
        """ Count the instances that books in the users history were recommended to the user by the LLM"""
        df['titles'] = df['user_id'].map(self.user_title_dict)
        df['baseline_rec_count'] = df['baseline_titles'].apply(lambda row: len(row['baseline_titles']))
        df['steered_rec_count'] = df['steered_titles'].apply(lambda row: len(row['steered_titles']))
        # Find items that were recommended to the user that are in their history (but not in initial prompt)
        df['baseline_count'] = df[['titles','baseline_titles','hist_titles']].apply(lambda row: len(list(set(row['titles']).intersection(row['baseline_titles']).difference(row['hist_titles']))), axis=1)
        df['steered_count'] = df[['titles','steered_titles','hist_titles']].apply(lambda row: len(list(set(row['titles']).intersection(row['steered_titles']).difference(row['hist_titles']))), axis=1)

    def precision(self, df, k=5):
        """ Calculates and prints the precision of the recs """
        baseline_count = df['baseline_count'].sum()
        baseline_denom_count = df['baseline_rec_count'].sum()
        steered_count = df['steered_count'].sum()
        steered_denom_count = df['steered_rec_count'].sum()
        bigger_denom = len(df) * k

        print(
        'Precision @ k: \n Baseline: ', 
         baseline_count/baseline_denom_count, 
         ", ",
        baseline_count/bigger_denom,
        "\n Steered: ", 
        steered_count/steered_denom_count, 
        ", ",
        steered_count/bigger_denom
        )

    def user_gender(self):
        for k,v in self.outputs.items():
            v['user_gender'] = self.map_dicts.gender_dict[str(k)]     

    def author_gender(self):
        for k,v in self.outputs.items():
            baseline_gender_list = []
            steered_gender_list = []
            for book in v['baseline_ids']:
                author_id = int(self.map_dicts.book_dict[book]['authors'][0]['author_id'])
                if author_id in self.map_dicts.author_gender['authorid']:
                    try:
                        baseline_gender_list.append(self.map_dicts.author_gender[self.map_dicts.author_gender.authorid==author_id].gender.values[0])
                    except:
                        pass
            for book in v['steered_ids']:
                author_id = int(self.map_dicts.book_dict[book]['authors'][0]['author_id'])
                if author_id in self.map_dicts.author_gender.authorid:
                    try:
                        steered_gender_list.append(self.map_dicts.author_gender[self.map_dicts.author_gender.authorid==author_id].gender.values[0])
                    except:
                        pass

            v['baseline_gender_list'] = baseline_gender_list
            v['steered_gender_list'] = steered_gender_list

    def author_gender_count(self, df):
        # sum author genders by baseline and steered
        df['bf_count'] = df['baseline_gender_list'].apply(lambda row: sum(1 for x in row if x =='female'))
        df['bm_count'] = df['baseline_gender_list'].apply(lambda row: sum(1 for x in row if x =='male'))
        df['bu_count'] = df['baseline_gender_list'].apply(lambda row: sum(1 for x in row if x =='unknown'))

        df['sf_count'] = df['steered_gender_list'].apply(lambda row: sum(1 for x in row if x =='female'))
        df['sm_count'] = df['steered_gender_list'].apply(lambda row: sum(1 for x in row if x =='male'))
        df['su_count'] = df['steered_gender_list'].apply(lambda row: sum(1 for x in row if x =='unknown'))


        print(
            " Female : Unknown : Male \n ",
            df['bf_count'].sum()/df['baseline_count'].sum(), 
            df['bm_count'].sum()/df['baseline_count'].sum(), 
            df['bu_count'].sum()/df['baseline_count'].sum() 
            "\n",
            df['sf_count'].sum()/df['steered_count'].sum(), 
            df['sm_count'].sum()/df['steered_count'].sum(), 
            df['su_count'].sum()/df['steered_count'].sum()
        )

if __name__ == '__main__':
    rec_data = RecData('goodreads_data',)
    llm_recs = LLMRecs('goodreads_data', rec_data.merged(), rec_data.user_title_dict())
            














