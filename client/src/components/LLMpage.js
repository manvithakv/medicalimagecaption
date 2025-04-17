import React, { useState } from 'react';
import axios from 'axios';
import './DLModelPage.css';

const LLMpage = () => {
    const [image, setImage] = useState(null);
    const [caption, setCaption] = useState('');
    const [preview, setPreview] = useState(null);

    const handleImageChange = (e) => {
        const file = e.target.files[0];
        setImage(file);
        setPreview(URL.createObjectURL(file));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!image) {
            alert('Please select an image.');
            return;
        }

        const formData = new FormData();
        formData.append('file', image);

        try {
            const res = await axios.post('http://127.0.0.1:5000/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setCaption(res.data.caption);
        } catch (err) {
            console.error(err);
        }
    };

   /* return (
        <div className="App">
            <h1>Image Captioning</h1>
            <form onSubmit={handleSubmit}>
                <input type="file" accept="image/*" onChange={handleImageChange} />
                <button type="submit">Upload and Predict</button>
            </form>
            {caption && <p>Caption: {caption}</p>}
        </div>
    );*/

    return (
        <div className="container">
            <h1>Image Captioning</h1>
            <form onSubmit={handleSubmit}>
                <input type="file" onChange={handleImageChange} />
                <button type="submit">Upload and Predict</button>
            </form>
            {preview && (
                <div className='preview'>
                    <h2>Preview:</h2>
                    <img src={preview} alt="Image Preview" style={{ maxWidth: '30%', height: '30' }} />
                </div>
            )}
            {caption && <div className='caption'>Caption: {caption}</div>}
        </div>
    );
  
}

export default LLMpage