import requests

url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=csv"
response = requests.get(url)
#Fetching the data in text form
data = response.content

#Saving the data in a csv file
with open("sats.csv","wb") as file:
    file.write(data)

print("Data Saved Successfully!")


