import React ,{useState} from 'react'
import axios from 'axios';
import './DLModelPage.css';

const DLModelPage = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [error, setError] = useState(null);
    const [preview, setPreview] = useState(null);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        setSelectedFile(file);
        setPreview(URL.createObjectURL(file));
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        const formData = new FormData();
        formData.append('image', selectedFile);
 
        try {
            const response = await axios.post('http://127.0.0.1:5002/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            setPrediction(response.data.prediction);
            setError(null);
        } catch (error) {
            console.error('Error uploading the file:', error);
            setError('Error uploading the file');
            setPrediction(null);
        }
    };

  /*return (
    <div >
            <h1>Upload chest X-ray for Effusion Detection</h1>
            <form onSubmit={handleSubmit}>
                <input type="file" onChange={handleFileChange} />
                <button type="submit">Upload and Predict</button>
            </form>
            {error && <div style={{ color: 'red' }}>{error}</div>}
            {prediction && <div>Prediction: {prediction}</div>}
    </div>
  )*/
    return (
        <div className="container">
            <h1>Upload chest X-ray for Effusion Detection</h1>
            <form onSubmit={handleSubmit}>
                <input type="file" onChange={handleFileChange} />
                <button type="submit">Upload and Predict</button>
            </form>
            {error && <div className='error'>{error}</div>}
            {preview && (
                <div className='preview'>
                    <h2>Preview:</h2>
                    <img src={preview} alt="Image Preview" style={{ maxWidth: '30%', height: '30' }} />
                </div>
            )}
            {prediction && <div className='caption'>Prediction: {prediction}</div>}
        </div>
    );

  
}

export default DLModelPage
