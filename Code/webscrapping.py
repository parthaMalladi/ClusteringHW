#No need to run this notebook because  Yummly API has been shut down in public.
import requests
import json
import pandas as pd
import numpy as np
import time

def initial_df(cykey, scuisine1, scourse, sresults):
    '''Use one cuisine to request and build the initial dataframe'''
    url = f"http://api.yummly.com/v1/api/recipes?{cykey}{scuisine1}{scourse}{sresults}"
    r = requests.get(url).json()

    print(r.keys())  # Fixed syntax
    print("Total Matches:", r.get('totalMatchCount', 0))
    print("Number of Recipes Fetched:", len(r.get('matches', [])))

    if 'matches' not in r:
        print("No matches found. Check API credentials.")
        return pd.DataFrame()

    # Build the initial DataFrame
    yum = pd.DataFrame(r['matches'])

    # Parse cuisine and course from 'attributes'
    yum['cuisine'] = yum['attributes'].apply(lambda x: x.get('cuisine', ['Unknown'])[0])
    yum['course'] = yum['attributes'].apply(lambda x: x.get('course', ['Unknown'])[0])
    print("Cuisine Counts:\n", yum['cuisine'].value_counts())

    # Filter for 'Chinese' cuisine only
    yum = yum[yum['cuisine'] == 'Chinese']
    
    return yum

def grow_df(cuisinedic, cykey, scourse, sresults, yum):
    '''Request recipes for all cuisines in cuisinedic, merge with existing dataframe'''
    for cs in cuisinedic:
        scuisine = f'&allowedCuisine[]=cuisine^cuisine-{cs}'
        url = f"http://api.yummly.com/v1/api/recipes?{cykey}{scuisine}{scourse}{sresults}"
        r = requests.get(url).json()
        
        print("Fetching:", scuisine)
        
        if 'matches' not in r:
            print(f"No recipes found for {cs}. Skipping.")
            continue

        newrecipes = pd.DataFrame(r['matches'])

        # Extract cuisine and course
        course = [item['attributes'].get('course', ['Unknown'])[0] for item in r['matches']]
        cuisine = [item['attributes'].get('cuisine', ['Unknown'])[0] for item in r['matches']]
        newrecipes['course'] = course
        newrecipes['cuisine'] = cuisine

        # Filter by correct cuisine
        newrecipes = newrecipes[newrecipes['cuisine'] == cuisinedic[cs]]

        # Merge with main dataset
        yum = pd.concat([yum, newrecipes], axis=0, ignore_index=True)
        print("Updated Dataset Shape:", yum.shape)
        
        # Sleep to prevent API rate limits (Remove if unnecessary)
        time.sleep(5)  # Change from 5 min to 5 sec for testing
    
    return yum

if __name__ == '__main__':
    # You need actual API credentials here!
    cykey = '_app_id=YOUR_APP_ID&_app_key=YOUR_APP_KEY'
    
    scuisine1 = '&allowedCuisine[]=cuisine^cuisine-chinese'
    scourse = '&allowedCourse[]=course^course-Main Dishes'
    sresults = '&maxResult=500'

    # Build initial DataFrame
    yum = initial_df(cykey, scuisine1, scourse, sresults)

    # Cuisine Mapping Dictionary
    cuisinedic = {
        'italian': 'Italian', 'mexican': 'Mexican', 'southern': 'Southern & Soul Food',
        'french': 'French', 'southwestern': 'Southwestern', 'indian': 'Indian',
        'cajun': 'Cajun & Creole', 'english': 'English', 'mediterranean': 'Mediterranean',
        'greek': 'Greek', 'spanish': 'Spanish', 'german': 'German', 'thai': 'Thai',
        'moroccan': 'Moroccan', 'irish': 'Irish', 'japanese': 'Japanese',
        'cuban': 'Cuban', 'swedish': 'Swedish', 'hungarian': 'Hungarian',
        'portuguese': 'Portuguese', 'american': 'American'
    }

    # Grow DataFrame with multiple cuisines
    yum = grow_df(cuisinedic, cykey, scourse, sresults, yum)

    # Save as pickle
    yum.to_pickle('data/yummly.pkl')

    print("Scraping Completed! Data saved as 'yummly.pkl'.")
