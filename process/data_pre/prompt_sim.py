prompt1 = """
###Task###
You are an expert in the field of Re-Identification (ReID). The existing description of the image is <caption>. Please refer to the following requirements to understand the general and specific features of the image, and then output the matching degree between the provided image and description. At the same time, answer the reason for this score. Note that the range of match score is from 0 to 9.
###Output###
1、Match score: XXX; 
2、Reason: XXX;
"""

prompt = """
###Task###
You are an expert in the field of Re-Identification (ReID). The three descriptions of the image are as follows:
1、{caption1};
2、{caption2};
3、{caption3};
Please refer to the following requirements to understand the general and specific features of the most prominent parts of the image, and then output the matching degree between three descriptions and the image in sequence. At the same time, provide a concise and clear reason for each score, At the same time, provide a concise and clear reason for each score, limit to 20 words or less. Note that the range of matching score is from 0.00 to 9.99. For example, the matching score is 5.65. In addition, the degree of matching only depends on whether the content described is correct, and the three descriptions are independent of each other.
###Output###
1、matching score: XXX; Reason: XXX;
2、matching score: XXX; Reason: XXX;
3、matching score: XXX; Reason: XXX;
"""