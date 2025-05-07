import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename_script = fileURLToPath(import.meta.url);
const __dirname_script = path.dirname(__filename_script);

export function cleanupOldQRCodes(tokenId) {
    const qrDir = path.resolve(__dirname_script, "../qr_codes");
    if (!fs.existsSync(qrDir)) {
        fs.mkdirSync(qrDir, { recursive: true });
        return;
    }
    
    const files = fs.readdirSync(qrDir);
    let cleanedCount = 0;
    
    files.forEach(file => {
        if (file.startsWith(`token_${tokenId}_`) && file.endsWith('.png')) {
            try {
                fs.unlinkSync(path.join(qrDir, file));
                cleanedCount++;
            } catch (error) {
                console.warn(`⚠️ Failed to delete old QR code ${file}: ${error.message}`);
            }
        }
    });
    
    if (cleanedCount > 0) {
        console.log(`🧹 Cleaned up ${cleanedCount} old QR code(s) for token ${tokenId}`);
    }
}
