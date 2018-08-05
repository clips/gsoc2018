# Google Summer of Code 2018
Kindly refer to [this](https://summerofcode.withgoogle.com/organizations/4798950528253952/) page for all the projects being undertaken by CLiPS in GSoC 2018.
## Multi Pronged Approach to Text Anonymization
### Architecture and Setup Instructions: [Click Here](https://github.com/clips/gsoc2018/tree/master/gdpr)
### Usage Guide: [Click Here](https://github.com/clips/gsoc2018/tree/master/gdpr/USAGE_GUIDE)
### GSoC Project Page : [Click Here](https://summerofcode.withgoogle.com/projects/#6562665841819648)
### About the Project
Text Anonymization refers to the processing of text, stripping it of any attributes/identifiers thus hiding sensitive details and protecting the identity of users.

This project consists of two principal parts, entity/identifier recognition, and the subsequent anonymization. First sensitive chunks of texts will be identified using various approaches including Named Entity Recognition, Regular Expression based pattern matching and TF-IDF based rare token detection. On being identified, the sensitive attributes will either be suppressed, generalized or deleted/replaced. Some of the approaches for generalization include Word Vector based obfuscation and usage of part holonyms.

This system is tied on top of a Django web-app. The system is provided with a dashboard where users can map attributes to the appropriate action and configure them. The system also has accesibilty features like RESTful API based anonymization end-points, Token Level Anonymization detail API, GUI based and API based anonymization of uploaded files etc. 

This system will provide a seamless, end-to-end solution for a firm's/user's text anonymization needs.

## Acknowledgements
#### Frontend : We are using Colorlib's [Bootstrap Admin Panel Template](https://colorlib.com/polygon/sufee/index.html) for our dashboard. Link : 	https://github.com/puikinsh/sufee-admin-dashboard
