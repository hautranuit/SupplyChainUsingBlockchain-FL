import crypto from 'crypto';

// create AES-256 key(32 byte)
const aesSecretKey = crypto.randomBytes(32).toString('hex');
console.log('AES Secret Key:', aesSecretKey);

// create HMAC key (32 byte)
const hmacSecretKey = crypto.randomBytes(32).toString('hex');
console.log('HMAC Secret Key:', hmacSecretKey);
