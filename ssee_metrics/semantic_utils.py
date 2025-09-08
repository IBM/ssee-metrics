from abc import ABC, abstractmethod

class SemanticUtils(ABC):
    """An abstract base class designed to provide a template for semantic utility operations:
        - encode a given entity into a semantic representation;
        - compute the semantic similarity between two encoded entities;
        - check if a given threshold value is valid
    """
    @abstractmethod
    def sem_enc(self, entity):
        pass

    @abstractmethod
    def sem_sim(self, encoded_e1, encoded_e2):
        pass

    @abstractmethod
    def threshold_check(self, T):
        pass

class SentenceTransformerSemanticUtils(SemanticUtils):
    """This class contains specific implementations for the methods defined in 
    the base class SemanticUtils, utilising the sentence_transformers library 
    for semantic encoding and similarity computation."""
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer, util
        self.model = SentenceTransformer(model_name)
        self.util = util

    def sem_enc(self, entity):
        """Encode the entity using the pre-trained model.

        Args:
            entity (str): The entity to encode.

        Returns:
            The encoded entity.
        """
        return self.model.encode(entity)

    def sem_sim(self, encoded_e1, encoded_e2):
        """Compute cosine similarity between the encoded entities.

        Args:
            encoded_e1: The first encoded entity.
            encoded_e2: The second encoded entity.

        Returns:
            float: The cosine similarity between the two encoded entities.
        """
        similarity = self.util.pytorch_cos_sim(encoded_e1, encoded_e2).item()
        return similarity

    def threshold_check(self, T):
        """Check if the threshold is a valid value.

        Args:
            T (float): The threshold value to check.
        Returns:
            bool: True if the threshold is valid
        Raises:
            ValueError: If the threshold is not valid (i.e. not between -1.0 and 1.0).
        """
        if T >= -1.0 and T <= 1.0:
            return True
        else:
            raise ValueError(f"Error: Threshold {T} should be between -1.0 and 1.0.")