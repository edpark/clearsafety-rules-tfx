from google.cloud import aiplatform as vertex_ai

import logging

def _get_source_query(bq_dataset_name, bq_table_name, ml_use, limit=None):
    query = f"""
    SELECT 
        IF(trip_month IS NULL, -1, trip_month) trip_month,
        IF(trip_day IS NULL, -1, trip_day) trip_day,
        IF(trip_day_of_week IS NULL, -1, trip_day_of_week) trip_day_of_week,
        IF(trip_hour IS NULL, -1, trip_hour) trip_hour,
        IF(trip_seconds IS NULL, -1, trip_seconds) trip_seconds,
        IF(trip_miles IS NULL, -1, trip_miles) trip_miles,
        IF(payment_type IS NULL, 'NA', payment_type) payment_type,
        IF(pickup_grid IS NULL, 'NA', pickup_grid) pickup_grid,
        IF(dropoff_grid IS NULL, 'NA', dropoff_grid) dropoff_grid,
        IF(euclidean IS NULL, -1, euclidean) euclidean,
        IF(loc_cross IS NULL, 'NA', loc_cross) loc_cross"""
    if ml_use:
        query += f""",
        tip_bin
    FROM {bq_dataset_name}.{bq_table_name} 
    WHERE ML_use = '{ml_use}'
    """
    else:
        query += f"""
    FROM {bq_dataset_name}.{bq_table_name} 
    """
    if limit:
        query += f"LIMIT {limit}"

    return query


def get_training_source_query(
    project, region, dataset_display_name, ml_use, limit=None
):
    vertex_ai.init(project=project, location=region)
    
    dataset = vertex_ai.TabularDataset.list(
        filter=f"display_name={dataset_display_name}", order_by="update_time"
    )[-1]
    bq_source_uri = dataset.gca_resource.metadata["inputConfig"]["bigquerySource"][
        "uri"
    ]
    _, bq_dataset_name, bq_table_name = bq_source_uri.replace("g://", "").split(".")

    return _get_source_query(bq_dataset_name, bq_table_name, ml_use, limit)


def get_serving_source_query(bq_dataset_name, bq_table_name, limit=None):

    return _get_source_query(bq_dataset_name, bq_table_name, ml_use=None, limit=limit)

def get_training_csv_source(gcs_location):
    logging.info(f'get_training_csv_source - gcs_location: {gcs_location}')
    if gcs_location.endswith('/'):
        return f'{gcs_location}data'
    else:
        return f'{gcs_location}/data'