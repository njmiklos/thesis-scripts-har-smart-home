from data_processing.compress_df_for_fm import Window


class ExtendedWindow(Window):
    """
    Extends 'Window' class with an attribute 'tokens' and a function 'update'.

    Attributes:
        true_annotation (str): True annotation for the window of data.
        data (str): The formatted window data or a result of its processing by a model.
        processing_time_ms (float): Total processing time of the window.
        max_memory_kb (float): The highest memory usage (kB) observed during the window processing.
        tokens (int): Total number of tokens used for the window at every stage.
    """
    def __init__(self, true_annotation: str, data: str, processing_time_ms: float = 0, 
                 max_memory_kb: float = 0, tokens: int = 0) -> None:
        super().__init__(true_annotation, data, processing_time_ms, max_memory_kb)

        if tokens < 0:
            raise ValueError(f'Tokens must be larger than 0, got {tokens}.')
        self.tokens = tokens

    def update(self, other: 'ExtendedWindow'):
        """
        Merges two windows' data by creating totals.
        """
        if self.data != other.data:
            self.data = other.data
        self.processing_time_ms += other.processing_time_ms
        if other.max_memory_kb > self.max_memory_kb:
            self.max_memory_kb = other.max_memory_kb
        self.tokens += other.tokens

    def to_dictionary(self) -> dict:
        """
        Returns the data about the object as a dictionary.

        Returns:
            dict: A dictionary representation of a ExtendedWindow object.
        """
        window_dict = super().to_dictionary()
        window_dict['tokens'] = self.tokens
        return window_dict