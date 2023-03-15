#############################################
# SMOKE DETECTOR V2.0 "SMOKY"
#
# Author: Daniel Eldan R.
# Date  : 12-2022
# Mail  : deldanr@gmail.com
# Name  : Tweet
# Desc  : Login the Twiteer API
############################################

#
# IMPORT BASE LIBRARIES
#
import tweepy
 
twitter_auth_keys = {
        "consumer_key"        : "---",
        "consumer_secret"     : "---",
        "access_token"        : "-------",
        "access_token_secret" : "---"
    }
 
auth = tweepy.OAuthHandler(
            twitter_auth_keys['consumer_key'],
            twitter_auth_keys['consumer_secret']
            )
auth.set_access_token(
            twitter_auth_keys['access_token'],
            twitter_auth_keys['access_token_secret']
            )
api = tweepy.API(auth)
