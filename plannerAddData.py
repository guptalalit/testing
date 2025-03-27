import os
import utils.datapreparation as datapreparation

class DataUpload:
    def __init__(self):
        num_attempts = 3

    def sequence(self, dataPath,new_app) -> str:
        """
        Execute a sequence of attempts to get a valid answer using RAG and validation.

        Args:
            parser: Input data or query to be processed.

        Returns:
            str: The validated answer or a default message if no valid answer is found.
        """
        datapreparation.execute_data(dataPath,new_app)