import AWS from 'aws-sdk';

export const sendMail = async (to: string, subject: string, body: string) => {
  // Check if running in development mode without AWS credentials
  if (process.env.NODE_ENV === 'development' && !process.env.AWS_ACCESS_KEY_ID) {
    console.log('📧 [DEV MODE] Email would be sent:');
    console.log(`   To: ${to}`);
    console.log(`   Subject: ${subject}`);
    console.log(`   Body: ${body}`);
    return Promise.resolve({ MessageId: 'dev-mode-email' });
  }

  const ses = new AWS.SES({
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
    region: process.env.AWS_REGION,
  });

  const params = {
    Source: process.env.EMAIL_FROM,
    Destination: {
      ToAddresses: [to],
    },
    Message: {
      Subject: {
        Data: subject,
      },
      Body: {
        Text: {
          Data: body,
        },
      },
    },
  };
  return ses.sendEmail(params).promise();
};
