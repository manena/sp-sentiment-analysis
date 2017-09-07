
# coding: utf-8

# In[ ]:

# We import Python's json library, Flask parts we're going to use and numpy
import json
from flask import Flask, request, Response
import numpy as np

# We create our Flask application:
app = Flask(__name__)

# We define our first url
@app.route("/", methods=["GET"])
def serve_html():
    """
    Simple function with the html and javascript that Flask is going to send to the browser
    """
    html_code = """
<html lang="es">
  <head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  </head>
    <body>
     <h1>Sentiment Analysis!</h1>
    
     <!-- Form for the phrase -->
     <form id="form1" action="">
        <p>Introduzca su frase en castellano / Write your phrase here (in Spanish):</p>
        <input id="phrase" type="text" cols="40" rows="5" value="">
        <br><br>
        <input type="submit" value="Check!">
    </form>
    <!-- Happy/sad face image that will appear with the first comment -->
    <img id="face" width="90" src="" style="display:none"/>
    <!-- The model result will appear here -->
    <p id="result"></p>

    <!-- Javascript (JQuery) -->
    <script src="https://code.jquery.com/jquery-3.2.1.min.js"></script>
    <script type="text/javascript">

    // Submit the form event:
    $('#form1').submit(function(event){
      event.preventDefault(); // So the browser won't refresh

      // AJAX request to Flask:
      $.ajax({url:"/predict", // When we submit the form, the following url will be called
                               // a http://server/predict
              method:"POST",
              contentType:"application/json",
              // We send a json with the phrase
              data: JSON.stringify({"phrase": $("#phrase").val()}),
              dataType:"json",
              
              success: function(result){
                  console.log(JSON.stringify(result))
                // A happy or sad face will be shown depending on the results
                if (result.prediction == "positive") {
                    $("#face").attr("src","http://freevector.co/wp-content/uploads/2012/04/51838-happy-smiling-emoticon-face.png");
                } else {
                    $("#face").attr("src","http://freevector.co/wp-content/uploads/2012/03/51850-sad-face.png");
                }
                // Show the face:
                $("#face").css("display","initial");
                // Show the result:
                $("#result").html("Result: "
                                     + result.positivity.toString());
            },
            error: function(result){
                console.log(JSON.stringify(result));
            }
          })
    });
    </script>
    </body>
    </html>"""
    
    # Send the result to the browser:
    return Response(response=html_code,
                    status=200,
                    content_type="text/html")



from nltk.tokenize import RegexpTokenizer
our_tokenizer = RegexpTokenizer("[\w']+")
# Our stemmer...
from nltk.stem.snowball import SpanishStemmer
stemmer_castellano = SpanishStemmer()
def tokenizer_stemmer(document):
    return [stemmer_castellano.stem(token) for token in our_tokenizer.tokenize(document)]
import os
my_dir = os.path.dirname(__file__)

# Flask url that will be executed when we call predict
@app.route("/predict", methods=["POST"])
def predict():

    # We unpickle the trained model
    import joblib
    filename = os.path.join(my_dir, 'best_model.pkl')    
    with open(filename, 'rb') as fo:
        my_model = joblib.load(fo)

    # We unpickle the vectorizer to treat data before predicting with the model 
    filename = os.path.join(my_dir, 'prepare_data_no_tokenizer.pkl')
    with open(filename, 'rb') as fo:
        my_vectorizer = joblib.load(fo)


    # We get the ajax Json
    datos_entrada = request.get_json()


    # We get the phrase
    phrase = datos_entrada['phrase']
    # We tokenize the phrase
    phrase = tokenizer_stemmer(phrase)
    phrase = [" ".join(str(x) for x in phrase)]

    # We vectorize the input data (phrase)
    my_vector = my_vectorizer.transform(phrase)
    
    # we don't have a predict_prob function in the model we have used, so we use decision_function
    # This doesn't giva a probability, but a value that can be positive or negative. 
    # The middle point should be around 0
    positivity = my_model.decision_function(my_vector)
    positivity = positivity[0]
    
    # Here we check how positive the phrase is. We use a cutting point below 0 because the model tends to
    # give a score that is lower than expected
    if positivity > -0.27:
        prediction = "positive"
    else:
        prediction = "negative"

    # We return the resulting Json:
    resp = {"positivity": positivity,
                 "prediction": prediction}

    return Response(response=json.dumps(resp),
                    status=200,
                    content_type="application/json")

# We run the application (in pythonanywhere.com we don't use this)
#if __name__ == "__main__":
#    app.run()

