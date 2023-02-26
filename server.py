from flask import Flask, request

import pandas as pd
import pickle

# Load in appropriate DataFrames, user ratings
articles_df = pd.read_csv('./articles.csv.zip', index_col='article_id')

# Customer data for collabortive filtering
df_customer = pd.read_csv('./df_customer.csv.zip', index_col='customer_id')

# Import final collab model
collab_model = pickle.load(open('./collaborative_model.sav', 'rb'))


app = Flask(__name__)


# create a simple route
@app.route('/recommend', methods=['GET'])
def recommend():
    most_common_values= df_customer['article_id'].value_counts().head(5)
    first_five_values = most_common_values.index.tolist()[:5]
    return {"data":first_five_values}


# create a route that accepts POST requests
@app.route('/recommend', methods=['POST'])
def postrecommend():
    data = request.json
    if not data:
        return None
    customer = data["customer_id"]
    n_recs = int(data["n_recs"])
    if customer and n_recs:
        # return customer
        have_bought = list(df_customer.loc[customer, 'article_id'])
        not_bought = articles_df.copy()
        # [not_bought.drop(x, inplace=True) for x in have_bought]
        not_bought.drop(have_bought, inplace=True)
        not_bought.reset_index(inplace=True)
        not_bought['est_purchase'] = not_bought['article_id'].apply(lambda x: collab_model.predict(customer, x).est)
        return not_bought.head(n_recs).to_dict()
    else:
        return None


if __name__ == '__main__':
    app.run(debug=True)
