import QRCode from 'qrcode';
import crypto from 'crypto';
import * as dotenv from 'dotenv';
import { mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

dotenv.config();

// AES encryption (AES-256-CBC with IV)
function encrypt(text, secretKey) {
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipheriv('aes-256-cbc', Buffer.from(secretKey, 'hex'), iv);
    let encrypted = cipher.update(text, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    return iv.toString('hex') + ':' + encrypted;
}

// Generate HMAC for integrity check
function generateHMAC(encryptedText, hmacKey) {
    return crypto.createHmac('sha256', Buffer.from(hmacKey, 'hex')).update(encryptedText).digest('hex');
}

// Save QR code as PNG
async function generateQRCode(finalPayload, outputPath) {
    await QRCode.toFile(outputPath, finalPayload, {
        color: { dark: '#000', light: '#fff' },
        width: 300
    });
    console.log(`✅ Encrypted QR code saved to: ${outputPath}`);
}

// Main function
async function main() {
    const ipfsCID = process.argv[2]; // IPFS CID as argument
    if (!ipfsCID) {
        console.error('❌ Please provide IPFS CID as an argument.');
        process.exit(1);
    }

    const secretKey = process.env.AES_SECRET_KEY;
    const hmacKey = process.env.HMAC_SECRET_KEY;

    if (!secretKey || secretKey.length !== 64) {
        console.error('❌ Missing or invalid AES_SECRET_KEY in .env (must be 32 bytes hex = 64 chars)');
        process.exit(1);
    }

    if (!hmacKey || hmacKey.length !== 64) {
        console.error('❌ Missing or invalid HMAC_SECRET_KEY in .env (must be 32 bytes hex = 64 chars)');
        process.exit(1);
    }

    // Encrypt the CID
    const encryptedCID = encrypt(ipfsCID, secretKey);

    // Generate HMAC for integrity verification
    const hmac = generateHMAC(encryptedCID, hmacKey);

    // Final payload: iv:encrypted:hmac
    const finalPayload = `${encryptedCID}:${hmac}`;

    // Output QR code to file
    const __dirname = dirname(fileURLToPath(import.meta.url));
    const qrOutputDir = join(__dirname, 'qr_codes');
    mkdirSync(qrOutputDir, { recursive: true });

    const qrFilePath = join(qrOutputDir, `qr_${Date.now()}.png`);
    await generateQRCode(finalPayload, qrFilePath);
}

main();
