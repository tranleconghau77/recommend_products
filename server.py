from flask import Flask
from flask_restful import reqparse, Api, Resource

import pandas as pd
import pickle

# Load in appropriate DataFrames, user ratings
articles_df = pd.read_csv('./articles.csv.zip', index_col='article_id')

# Customer data for collabortive filtering
df_customer = pd.read_csv('./df_customer.csv', index_col='customer_id')

# Import final collab model
collab_model = pickle.load(open('./collaborative_model.sav', 'rb'))

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('customer_id')
parser.add_argument('n_recs')


class Recommend(Resource):
    def get(self):
        return "Connect Successfully"
    def post(self):
        # customer = "0063faad1cf48d3d6e02d02b3890248173fd59aefe8ecd0c3817b9688ba93124"
        # n_recs = 5
        try:
            args = parser.parse_args()
            customer = args["customer_id"]
            n_recs = int(args["n_recs"])
            have_bought = list(df_customer.loc[customer, 'article_id'])
            not_bought = articles_df.copy()
            # [not_bought.drop(x, inplace=True) for x in have_bought]
            not_bought.drop(have_bought, inplace=True)
            not_bought.reset_index(inplace=True)
            not_bought['est_purchase'] = not_bought['article_id'].apply(lambda x: collab_model.predict(customer, x).est)
            not_bought.sort_values(by='est_purchase', ascending=False, inplace=True)
            
            not_bought.rename(columns={'prod_name':'Product Name', 'author':'Author',
                                    'product_type_name':'Product Type Name', 'product_group_name':'Product Group Name',
                                    'index_group_name':'Index Group Name', 'garment_group_name ':'Garment Group Name'}, inplace=True)
            not_bought = not_bought.iloc[:100, :]
            not_bought.drop(['product_code', 'product_type_no', 'graphical_appearance_no','graphical_appearance_name', 'colour_group_code', 'colour_group_name',
            'perceived_colour_value_id', 'perceived_colour_value_name','perceived_colour_master_id', 'perceived_colour_master_name',
            'department_no', 'department_name', 'index_code', 'index_name','index_group_no', 'section_no', 'section_name',
            'garment_group_no', 'detail_desc','est_purchase'], axis=1, inplace=True)
            # not_bought = not_bought[['article_id','Product Name', 'Product Type Name', 'Product Group Name', 'Index Group Name', 'Garment Group Name']]
            not_bought = not_bought.sample(frac=1).reset_index(drop=True)
            
            return not_bought.head(n_recs).to_dict()
        except:
            return "Error"


##
## Actually setup the Api resource routing here
##

api.add_resource(Recommend, '/recommend')

if __name__ == '__main__':
    app.run(debug=True)