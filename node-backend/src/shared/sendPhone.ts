import AWS from 'aws-sdk';

export const sendPhone = async (to: string, message: string) => {
  // Check if running in development mode without AWS credentials
  if (process.env.NODE_ENV === 'development' && !process.env.AWS_ACCESS_KEY_ID) {
    console.log('📱 [DEV MODE] SMS would be sent:');
    console.log(`   To: ${to}`);
    console.log(`   Message: ${message}`);
    return Promise.resolve({ MessageId: 'dev-mode-sms' });
  }

  const sns = new AWS.SNS({
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
    region: process.env.AWS_REGION,
  });

  const params = {
    Message: message,
    PhoneNumber: to,
  };
  return sns.publish(params).promise();
};
