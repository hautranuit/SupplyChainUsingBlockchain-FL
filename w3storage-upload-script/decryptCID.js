import crypto from 'crypto';
import * as dotenv from 'dotenv';

dotenv.config();

// Decrypt function (AES-256-CBC)
function decrypt(encryptedText, secretKey) {
    const [ivHex, encryptedHex] = encryptedText.split(':');
    const iv = Buffer.from(ivHex, 'hex');
    const encryptedBuffer = Buffer.from(encryptedHex, 'hex');

    const decipher = crypto.createDecipheriv('aes-256-cbc', Buffer.from(secretKey, 'hex'), iv);
    let decrypted = decipher.update(encryptedBuffer, null, 'utf8');
    decrypted += decipher.final('utf8');
    return decrypted;
}

// Verify HMAC
function verifyHMAC(encryptedText, hmac, hmacKey) {
    const expectedHMAC = crypto.createHmac('sha256', Buffer.from(hmacKey, 'hex')).update(encryptedText).digest('hex');
    return expectedHMAC === hmac;
}

function main() {
    const input = process.argv[2];
    if (!input) {
        console.error('❌ Please provide encrypted text as an argument');
        process.exit(1);
    }

    const secretKey = process.env.AES_SECRET_KEY;
    const hmacKey = process.env.HMAC_SECRET_KEY;

    if (!secretKey || secretKey.length !== 64) {
        console.error('❌ Missing or invalid AES_SECRET_KEY in .env');
        process.exit(1);
    }

    if (!hmacKey || hmacKey.length !== 64) {
        console.error('❌ Missing or invalid HMAC_SECRET_KEY in .env');
        process.exit(1);
    }

    try {
        // Split input: iv:encrypted:hmac
        const parts = input.split(':');
        if (parts.length !== 3) {
            throw new Error('Invalid encrypted input format. Expected format: iv:encrypted:hmac');
        }

        const [ivHex, encryptedHex, hmac] = parts;
        const encryptedText = `${ivHex}:${encryptedHex}`;

        // Verify HMAC
        if (!verifyHMAC(encryptedText, hmac, hmacKey)) {
            throw new Error('HMAC verification failed. Data integrity compromised.');
        }

        // Decrypt if HMAC is valid
        const decryptedCID = decrypt(encryptedText, secretKey);
        console.log('✅ Decrypted CID:', decryptedCID);
        console.log(`🔗 IPFS Gateway Link: https://${decryptedCID}.ipfs.w3s.link`);
    } catch (error) {
        console.error('❌ Decryption failed:', error.message);
    }
}

main();
