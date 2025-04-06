from typing import List

class MatcherPipeline:
    def __init__(self, matchers: List):
        """
        Initializes the pipeline with a list of instantiated matcher objects.

        :param matchers: List of matcher objects (e.g., ExactMatcher, ProximityMatcher)
        """
        self.matchers = matchers

    def run(self, text: str, output: str = "detailed") -> List[dict]:
        """
        Runs all the matchers on the text and collects the results.

        :param text: The input text to run the matchers on.
        :param output: The output format, default is "detailed".
        :return: A list of match results showing which matcher found which keywords.
        """
        results = []

        for matcher in self.matchers:
            matcher_name = matcher.__class__.__name__
            result = matcher.match(text, output)

            results.append({
                "matcher_name": matcher_name,
                "matches": result
            })

        return results