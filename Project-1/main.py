# Imports
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import flask
import logging
import os
import tfmodel
from google.cloud import bigquery
from google.cloud import storage

# Set up logging
logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(levelname)s - %(message)s',
                     datefmt='%Y-%m-%d %H:%M:%S')

PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT') 
logging.info('Google Cloud project is {}'.format(PROJECT))

# Initialisation
logging.info('Initialising app')
app = flask.Flask(__name__)

logging.info('Initialising BigQuery client')
BQ_CLIENT = bigquery.Client()

BUCKET_NAME = PROJECT + '-images'
logging.info('Initialising access to storage bucket {}'.format(BUCKET_NAME))
APP_BUCKET = storage.Client().bucket(BUCKET_NAME)

logging.info('Initialising TensorFlow classifier')
TF_CLASSIFIER = tfmodel.Model(
    app.root_path + "/static/tflite/model.tflite",
    app.root_path + "/static/tflite/dict.txt"
)
logging.info('Initialisation complete')

# End-point implementation
@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/classes')
def classes():
    results = BQ_CLIENT.query(
    '''
        Select Description, COUNT(*) AS NumImages
        FROM `bdcc-project1-417811.openimages.image_labels`
        JOIN `bdcc-project1-417811.openimages.classes` USING(Label)
        GROUP BY Description
        ORDER BY Description
    ''').result()
    logging.info('classes: results={}'.format(results.total_rows))
    data = dict(results=results)
    return flask.render_template('classes.html', data=data)

@app.route('/relations')
def relations():
    #Fetch relations
    relations_results = BQ_CLIENT.query(
    '''
        SELECT DISTINCT Relation, COUNT(Relation)
        FROM `bdcc-project1-417811.openimages.relations`
        GROUP BY Relation
        ORDER BY Relation
    '''
    ).result()

    logging.info('relations: results={}'.format(relations_results.total_rows))
    data = dict(relations_results=relations_results)
    return flask.render_template('relations.html', data=data)


@app.route('/image_info')
def image_info():
    image_id = flask.request.args.get('image_id')

    #Fetch classes
    query = '''
        SELECT Description
        FROM `bdcc-project1-417811.openimages.image_labels`
        JOIN `bdcc-project1-417811.openimages.classes` USING(Label)
        WHERE imageId = @imageId
        ORDER BY Description
    '''

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("imageId", "STRING", image_id),
        ]
    )
    
    classes_results = BQ_CLIENT.query(query, job_config=job_config).result()

    #Fetch relations
    query = '''
        SELECT c1.Description, Relation, c2.Description
        FROM bdcc-project1-417811.openimages.classes AS c1
        JOIN bdcc-project1-417811.openimages.relations ON c1.Label = Label1
        JOIN bdcc-project1-417811.openimages.classes AS c2 ON c2.Label = Label2 
        WHERE ImageId = @imageId
        GROUP BY c1.Description, Relation, c2.Description
    '''

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("imageId", "STRING", image_id),
        ]
    )
    
    relations_results = BQ_CLIENT.query(query, job_config=job_config).result()


    logging.info('classes: results={}'.format(classes_results.total_rows))
    logging.info('relations: results={}'.format(relations_results.total_rows))
    data = dict(classes_results=classes_results,relations_results=relations_results)
    return flask.render_template('image_info.html', image_id=image_id, data=data)


@app.route('/image_search')
def image_search():
    description = flask.request.args.get('description', default='')
    image_limit = flask.request.args.get('image_limit', default=10, type=int)
    #Fetch image search
    query = '''
        SELECT ImageId
        FROM bdcc-project1-417811.openimages.classes
        JOIN bdcc-project1-417811.openimages.image_labels USING (Label)
        WHERE Description = @description
        GROUP BY ImageId
        LIMIT @image_limit
    '''

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("description", "STRING", description),
            bigquery.ScalarQueryParameter("image_limit", "INTEGER", image_limit)
        ]
    )

    image_search_results = BQ_CLIENT.query(query, job_config=job_config).result()

    #Fetch image url
    ##TODO e preciso encher o bucket com as imagens, de momento esta vazio
    logging.info('image_search: results={}'.format(image_search_results.total_rows))
    data = dict(image_search_results=image_search_results)
    return flask.render_template('image_search.html', data=data, description=description, image_limit=image_limit)


@app.route('/relation_search')
def relation_search():
    class1 = flask.request.args.get('class1', default='%')
    relation = flask.request.args.get('relation', default='%')
    class2 = flask.request.args.get('class2', default='%')
    image_limit = flask.request.args.get('image_limit', default=10, type=int)
    #Fetch relation search
    query = '''
        SELECT ImageId, c1.Description, c2.Description
        FROM bdcc-project1-417811.openimages.relations
        JOIN bdcc-project1-417811.openimages.image_labels USING (ImageId)
        JOIN bdcc-project1-417811.openimages.classes c1 ON c1.Label = Label1 AND c1.Description LIKE CONCAT('%', @class1, '%')
        JOIN bdcc-project1-417811.openimages.classes c2 ON c2.Label = Label2 AND c2.Description LIKE CONCAT('%', @class2, '%')
        WHERE Relation = @relation AND @class1 <> "" AND @class2 <> "" 
        GROUP BY ImageId, c1.Description, c2.Description
        ORDER BY ImageId, c1.Description, c2.Description
        LIMIT @image_limit
    '''

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("class1", "STRING", class1),
            bigquery.ScalarQueryParameter("relation", "STRING", relation),
            bigquery.ScalarQueryParameter("class2", "STRING", class2),
            bigquery.ScalarQueryParameter("image_limit", "INTEGER", image_limit),
        ]
    )

    relation_search_results = BQ_CLIENT.query(query, job_config=job_config).result()
    #Fetch image url
    ##TODO
    logging.info('relation_search: results={}'.format(relation_search_results.total_rows))
    data = dict(relation_search_results=relation_search_results)
    return flask.render_template('relation_search.html', data=data, class1=class1, relation=relation, class2=class2, image_limit=image_limit)


@app.route('/image_classify_classes')
def image_classify_classes():
    with open(app.root_path + "/static/tflite/dict.txt", 'r') as f:
        data = dict(results=sorted(list(f)))
        return flask.render_template('image_classify_classes.html', data=data)
 
@app.route('/image_classify', methods=['POST'])
def image_classify():
    files = flask.request.files.getlist('files')
    min_confidence = flask.request.form.get('min_confidence', default=0.25, type=float)
    results = []
    if len(files) > 1 or files[0].filename != '':
        for file in files:
            classifications = TF_CLASSIFIER.classify(file, min_confidence)
            blob = storage.Blob(file.filename, APP_BUCKET)
            blob.upload_from_file(file, blob, content_type=file.mimetype)
            logging.info('image_classify: filename={} blob={} classifications={}'\
                .format(file.filename,blob.name,classifications))
            results.append(dict(bucket=APP_BUCKET,
                                filename=file.filename,
                                classifications=classifications))
    
    data = dict(bucket_name=APP_BUCKET.name, 
                min_confidence=min_confidence, 
                results=results)
    return flask.render_template('image_classify.html', data=data)



if __name__ == '__main__':
    # When invoked as a program.
    logging.info('Starting app')
    app.run(host='0.0.0.0', port=8080, debug=True)
